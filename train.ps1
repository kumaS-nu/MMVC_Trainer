[System.IO.FileInfo[]] $files = dir "configs" -File
mkdir "models"
./env/Scripts/Activate.ps1
foreach ($f in $files) {
    if ($f.Name.Contains("baseconfig")) {
        continue
    }
    $model_dir = "models" + $f.Name
    mkdir $model_dir
    $c = "configs/" + $f.Name
    python train_ms.py -c $c -m $model_dir -fg fine_model/G_180000.pth -fd fine_model/D_180000.pth
}