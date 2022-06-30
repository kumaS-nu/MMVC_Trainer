import os
import shutil
import wave
import numpy as np
import pyworld as pw


origin_path = 'dataset/origin/'
origin2file_dir = 'parallel100/wav24kHz16bit/'
text_path = 'dataset/transcripts_utf8.txt'
dst_path = 'dataset/textful/'

def load_transcripts():
    with open(text_path, encoding='utf-8')as f:
        texts = f.readlines()
    transcripts = {}
    for t in texts:
        splited = t.split(':')
        transcripts[splited[0]] = splited[1]
    return transcripts

def analysis_resynthesis(signal, f0_rate, sp_rate):

    sample_rate = 24000
    # 音響特徴量の抽出
    f0, t = pw.dio(signal, sample_rate)  # 基本周波数の抽出
    f0 = pw.stonemask(signal, f0, t, sample_rate)  # refinement
    sp = pw.cheaptrick(signal, f0, t, sample_rate)  # スペクトル包絡の抽出
    ap = pw.d4c(signal, f0, t, sample_rate)  # 非周期性指標の抽出

    # ピッチシフト
    modified_f0 = f0_rate * f0

    # フォルマントシフト（周波数軸の一様な伸縮）
    modified_sp = np.zeros_like(sp)
    sp_range = int(modified_sp.shape[1] * sp_rate)
    for f in range(modified_sp.shape[1]):
        if (f < sp_range):
            if sp_rate >= 1.0:
                modified_sp[:, f] = sp[:, int(f / sp_rate)]
            else:
                modified_sp[:, f] = sp[:, int(sp_rate * f)]
        else:
            modified_sp[:, f] = sp[:, f]

    # 再合成
    synth = pw.synthesize(modified_f0, modified_sp, ap, sample_rate)

    return synth

def convert(transcripts):
    files = os.listdir(origin_path)
    dirs = [f for f in files if os.path.isdir(os.path.join(origin_path, f))]
    for d in dirs:
        file_dirs = os.listdir(os.path.join(origin_path, d, origin2file_dir))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        jpv_dir = os.join(dst_path, d[-3:] + '_jpv')
        if not os.path.exists(jpv_dir):
            os.mkdir(jpv_dir)
        jpv_text = os.join(jpv_dir, 'text')
        if not os.path.exists(jpv_text):
            os.mkdir(jpv_text)
        jpv_wav = os.join(jpv_dir, 'wav')
        if not os.path.exists(jpv_wav):
            os.mkdir(jpv_wav)

        conv_dir1 = os.join(dst_path, str(int(d[-3]) + 1) + d[-2:] + '_jpv_conv10')
        if not os.path.exists(conv_dir1):
            os.mkdir(conv_dir1)
        conv_text1 = os.join(conv_dir1, 'text')
        if not os.path.exists(conv_text1):
            os.mkdir(conv_text1)
        conv_wav1 = os.join(conv_dir1, 'wav')
        if not os.path.exists(conv_wav1):
            os.mkdir(conv_wav1)

        conv_dir2 = os.join(dst_path, str(int(d[-3]) + 2) + d[-2:] + '_jpv_conv20')
        if not os.path.exists(conv_dir2):
            os.mkdir(conv_dir2)
        conv_text2 = os.join(conv_dir2, 'text')
        if not os.path.exists(conv_text2):
            os.mkdir(conv_text2)
        conv_wav2 = os.join(conv_dir2, 'wav')
        if not os.path.exists(conv_wav2):
            os.mkdir(conv_wav2)

        conv_dir3 = os.join(dst_path, str(int(d[-3]) + 3) + d[-2:] + '_jpv_conv30')
        if not os.path.exists(conv_dir3):
            os.mkdir(conv_dir3)
        conv_text3 = os.join(conv_dir3, 'text')
        if not os.path.exists(conv_text3):
            os.mkdir(conv_text3)
        conv_wav3 = os.join(conv_dir3, 'wav')
        if not os.path.exists(conv_wav3):
            os.mkdir(conv_wav3)

        for file in file_dirs:
            file_name = os.path.splitext(file)[0]
            dst_name = file_name[-3:]
            with open(os.path.join(jpv_text, dst_name + ".txt"), mode='w') as f:
                f.write(transcripts[file_name])
            with open(os.path.join(conv_text1, dst_name + ".txt"), mode='w') as f:
                f.write(transcripts[file_name])
            with open(os.path.join(conv_text2, dst_name + ".txt"), mode='w') as f:
                f.write(transcripts[file_name])
            with open(os.path.join(conv_text3, dst_name + ".txt"), mode='w') as f:
                f.write(transcripts[file_name])
            shutil.copy(os.path.join(origin_path, d, file), os.path.join(jpv_wav, dst_name + ".wav"))

            with wave.open(os.path.join(origin_path, d, file), 'r') as f:
                raw_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
                ch = f.getnchannels()
                width = f.getsampwidth()
                fr = f.getframerate()
            data = np.zeros((int((raw_data.size - 1) / 4096) + 1 ) * 4096, dtype=np.float64)
            data[:raw_data.size] = raw_data.astype(np.float64)
            data = data.reshape((int((raw_data.size - 1) / 4096) + 1, 4096))
            
            conv = []
            for d in data:
                conv.append(analysis_resynthesis(d, 1.19, 0.92))
            conv = np.array(conv).flatten().astype(np.int16)
            with wave.open(os.path.join(conv_wav1, dst_name + ".wav"), 'w') as f:
                f.setnchannels(ch)
                f.setsampwidth(width)
                f.setframerate(fr)
                f.writeframes(conv.tobytes())

            conv = []
            for d in data:
                conv.append(analysis_resynthesis(d, 1.41, 0.84))
            conv = np.array(conv).flatten().astype(np.int16)
            with wave.open(os.path.join(conv_wav2, dst_name + ".wav"), 'w') as f:
                f.setnchannels(ch)
                f.setsampwidth(width)
                f.setframerate(fr)
                f.writeframes(conv.tobytes())

            conv = []
            for d in data:
                conv.append(analysis_resynthesis(d, 1.68, 0.77))
            conv = np.array(conv).flatten().astype(np.int16)
            with wave.open(os.path.join(conv_wav3, dst_name + ".wav"), 'w') as f:
                f.setnchannels(ch)
                f.setsampwidth(width)
                f.setframerate(fr)
                f.writeframes(conv.tobytes())


if __name__ == '__main__':
    transcripts = load_transcripts()
    convert(transcripts)