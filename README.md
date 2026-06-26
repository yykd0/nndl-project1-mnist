# HW1 Fashion-MNIST MLP

Public repository for the computer vision homework: build a three-layer MLP classifier for Fashion-MNIST from scratch with NumPy.

## Links

- Model weights: https://drive.google.com/file/d/1JivAIdlZ-d-31_CWD0BR8a-r3GXphVN5/view?usp=drivesdk
- PDF report: https://drive.google.com/file/d/15sGfMvcHesSWj1FsOF-2868lh-2XLBhl/view?usp=drivesdk
- Full submission package: https://drive.google.com/file/d/1U2JA7R866qn2kzJ1h8KOXk5k1nJNlvGg/view?usp=drivesdk

## Result

The best NumPy-only MLP uses the following configuration:

- Architecture: `784 -> 128 -> 128 -> 10`
- Activation: ReLU
- Optimizer: SGD
- Initial learning rate: `0.12`
- Learning-rate decay: step decay every 4 epochs with gamma `0.5`
- Weight decay: `1e-4`
- Validation accuracy: `0.8774`
- Test accuracy: `0.8692`

## Repository Layout

The Fashion-MNIST homework files are under `HW1_FashionMNIST/`.

- `HW1_FashionMNIST/src/data.py`: dataset download, IDX parsing, normalization, train/validation split
- `HW1_FashionMNIST/src/layers.py`: Linear, ReLU, Sigmoid, Tanh, softmax cross-entropy, manual backward pass
- `HW1_FashionMNIST/src/model.py`: configurable three-layer MLP
- `HW1_FashionMNIST/src/engine.py`: training loop, validation, best-checkpoint saving
- `HW1_FashionMNIST/src/search.py`: hyperparameter grid search
- `HW1_FashionMNIST/src/test.py`: test-set accuracy and confusion matrix
- `HW1_FashionMNIST/src/report.py`: PDF report generation
- `HW1_FashionMNIST/results/metrics.json`: final metrics and confusion matrix

## Environment

```powershell
pip install -r HW1_FashionMNIST/requirements.txt
```

Only NumPy is used for neural-network computation. Pillow and reportlab are used for visualization and report generation.

## Train

```powershell
cd HW1_FashionMNIST
python -m src.train --epochs 8 --hidden-dim 128 --activation relu --lr 0.12 --weight-decay 0.0001 --batch-size 128 --lr-decay-step 4 --lr-decay-gamma 0.5
```

## Hyperparameter Search

```powershell
cd HW1_FashionMNIST
python -m src.search --quick
python -m src.search --epochs 4
```

## Test

Download the model weight file from the Drive link above and place it at `HW1_FashionMNIST/checkpoints/best_model.pkl`, then run:

```powershell
cd HW1_FashionMNIST
python -m src.test --model checkpoints/best_model.pkl
```

This prints test accuracy and the 10x10 confusion matrix.

## Report

```powershell
cd HW1_FashionMNIST
python -m src.report
```

The generated PDF is also available from the Drive link above.
