./env/Scripts/Activate.ps1
cd monotonic_align
python setup.py build_ext --inplace
cd ..

[System.IO.FileInfo[]] $files = dir "configs" -File

foreach ($f in $files) {
    if ($f.Name.Contains("baseconfig")) {
        continue
    }
    if ($f.Name.Contains("gitignore")){
        continue
    }
    $model_dir = "models/" + [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    mkdir $model_dir
    $c = "configs/" + $f.Name
    python train_ms.py -c $c -m $model_dir -fg fine_model/G_180000.pth -fd fine_model/D_180000.pth
}