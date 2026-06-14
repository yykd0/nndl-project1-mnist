# Neural Network and Deep Learning Project 2

姓名：邓凯源  
学号：22300680061

This folder contains the reproducible code for Project 2: CIFAR-10 classification and Batch Normalization analysis.

## Main files

- `project2_experiments.py`: runs the compact CIFAR-10 model sweep and VGG-A BatchNorm loss-landscape experiment.
- `models/vgg.py`: VGG-A and VGG-A with BatchNorm implementations.
- `data/loaders.py`: CIFAR-10 data loading helper with safe extraction and automatic download fallback.
- `utils/nn.py`: weight initialization helpers.
- `VGG_Loss_Landscape.py`: compatibility entry point for the BN loss-landscape experiment.
- `make_report_cn.py`: builds the Chinese PDF report from saved metrics and figures.
- `WEIGHTS_LINK.txt`: Google Drive link for the trained weights file.

## Reproduction

```bash
pip install -r requirements.txt
python project2_experiments.py --epochs 8 --train-items 12000 --test-items 2000 --bn-epochs 1 --bn-train-items 1024 --bn-test-items 512 --batch-size 128 --num-workers 0 --seed 2026
python make_report_cn.py --name "邓凯源" --student-id "22300680061" --code-link "https://github.com/yykd0/nndl-project1-mnist/tree/main/project2-cifar-bn" --dataset-link "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" --weights-link "https://drive.google.com/file/d/1rVHmbTbaL2_U4Iu9SYwPujOTP2oJ0uJK/view?usp=drivesdk"
```

## Links

- Code: https://github.com/yykd0/nndl-project1-mnist/tree/main/project2-cifar-bn
- Dataset: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- Trained weights: https://drive.google.com/file/d/1rVHmbTbaL2_U4Iu9SYwPujOTP2oJ0uJK/view?usp=drivesdk

## Notes

The Google Drive weights file is stored as `best_compact_cifar.pt.b64.txt` because the Drive connector available here cannot upload arbitrary `.pt` binary files directly. Decode the text file with base64 to recover `best_compact_cifar.pt`. The local submission package also includes the original `.pt` file.
