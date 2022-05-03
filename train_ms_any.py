import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import datetime
import pytz
from tqdm import tqdm
import warnings

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
  BypassEncoder,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0

#stftの警告対策
warnings.resetwarnings()
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,375,750,1125,1500,1875,2250,2625,3000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)

  net_b = BypassEncoder(
      hps.data.filter_length // 2 + 1,
      hps.model.inter_channels,
      hps.model.hidden_channels,
      5,
      1,
      16,
      hps.model.gin_channels).cuda(rank)
  optim_b = torch.optim.AdamW(
      net_b.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  #net_g = DDP(net_g, device_ids=[rank])
  #net_b = DDP(net_b, device_ids=[rank])

  logger.info('Loading : '+str(hps.model_g))
  utils.load_checkpoint(hps.model_g, net_g)
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "B_*.pth"), net_b, optim_b)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_b = torch.optim.lr_scheduler.ExponentialLR(optim_b, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_b], optim_b, scheduler_b, scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_b], optim_b, scheduler_b, scaler, [train_loader, None], None, None)
    scheduler_b.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_b = nets
  optim_b = optims
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_b.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader, desc="Epoch {}".format(epoch))):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (use_z, z_p, m_p, logs_p, m_q, use_logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
      #Bypass Encoder
      bypass_z, bypass_m, bypass_logs, bypass_y_mask = net_b(spec, spec_lengths)
      bypass_z_any = net_g.flow_(bypass_z, bypass_y_mask, speakers)
      bypass_z_slice, bypass_ids_slice = commons.rand_slice_segments(bypass_z_any, spec_lengths, hps.train.segment_size // hps.data.hop_length)
      bypass_out = net_g.dec_(bypass_z_slice)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, bypass_ids_slice, hps.train.segment_size // hps.data.hop_length)
    bypass_out = bypass_out.float()
    bypass_out_mel = mel_spectrogram_torch(
        bypass_out.squeeze(1), 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
    )

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, bypass_out_mel) * hps.train.c_mel
        loss_kl = kl_loss(use_z, use_logs_q, bypass_z, bypass_logs, z_mask) * hps.train.c_kl

        loss_gen_all = loss_mel + loss_kl
    optim_b.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_b)
    grad_norm_b = commons.clip_grad_value_(net_b.parameters(), None)
    scaler.step(optim_b)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_b.param_groups[0]['lr']
        losses = [loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info(datetime.datetime.now(pytz.timezone('Asia/Tokyo')))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_b": grad_norm_b}
        scalar_dict.update({"loss/g/mel": loss_mel, "loss/g/kl": loss_kl})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(bypass_out_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, net_b, eval_loader, writer_eval)
        utils.save_checkpoint(net_b, optim_b, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "B_{}.pth".format(global_step)))
    global_step += 1

 
def evaluate(hps, generator, net_b, eval_loader, writer_eval):
    generator.eval()
    net_b.eval()
    scalar_dict = {}
    scalar_dict.update({"loss/g/mel": 0.0, "loss/g/kl": 0.0})
    with torch.no_grad():
      #evalのデータセットを一周する
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(eval_loader, desc="Epoch {}".format("eval"))):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)
        #autocastはfp16のおまじない
        with autocast(enabled=hps.train.fp16_run):
          #Generator
          y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
          (z, z_p, m_p, logs_p, m_q, logs_q) = generator(x, x_lengths, spec, spec_lengths, speakers)

          bypass_z, bypass_m, bypass_logs, bypass_y_mask = net_b(spec, spec_lengths)
          bypass_z_any = generator.flow_(bypass_z, bypass_y_mask, speakers)
          bypass_z_slice, bypass_ids_slice = commons.rand_slice_segments(bypass_z_any, spec_lengths, hps.train.segment_size // hps.data.hop_length)
          bypass_out = generator.dec_(bypass_z_slice)

          mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
          y_mel = commons.slice_segments(mel, bypass_ids_slice, hps.train.segment_size // hps.data.hop_length)
        bypass_out = bypass_out.float()
        bypass_out_mel = mel_spectrogram_torch(
            bypass_out.squeeze(1), 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )
        batch_num = batch_idx

        with autocast(enabled=hps.train.fp16_run):
          with autocast(enabled=False):
            loss_mel = F.l1_loss(y_mel, bypass_out_mel) * hps.train.c_mel
            loss_kl = kl_loss(z, logs_q, bypass_z, bypass_logs, z_mask) * hps.train.c_kl

        scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] + loss_mel
        scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] + loss_kl
      
      #lossをepoch1周の結果をiter単位の平均値に
      scalar_dict["loss/g/mel"] = scalar_dict["loss/g/mel"] / (batch_num+1)
      scalar_dict["loss/g/kl"] = scalar_dict["loss/g/kl"] / (batch_num+1)

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
                           
if __name__ == "__main__":
  main()
