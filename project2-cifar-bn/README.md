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

## Reproduction

```bash
pip install -r requirements.txt
python project2_experiments.py --epochs 8 --train-items 12000 --test-items 2000 --bn-epochs 1 --bn-train-items 1024 --bn-test-items 512 --batch-size 128 --num-workers 0 --seed 2026
python make_report_cn.py --name "邓凯源" --student-id "22300680061" --code-link "https://github.com/yykd0/nndl-project1-mnist/tree/main/project2-cifar-bn" --dataset-link "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" --weights-link "<model weights link>"
```

## Notes

The CIFAR-10 dataset and trained weights are not stored in this source folder. The dataset is downloaded from the official CIFAR-10 URL, and model weights are provided separately in the submitted report.
