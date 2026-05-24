# Project 1 代码说明

本目录包含一个 NumPy-only 的 MNIST 手写数字分类实现。

## 已实现内容

- `mynn/op.py`
  - `Linear.forward` 与 `Linear.backward`
  - 数值稳定的 `MultiCrossEntropyLoss`
  - 自实现 `conv2D.forward` 与 `conv2D.backward`
  - 辅助层：`ReLU`、`Flatten`、`MaxPool2D`
- `mynn/models.py`
  - MLP 基线模型
  - CNN 模型：`Conv -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Linear`
- `mynn/optimizer.py`
  - SGD
  - MomentumGD
- `mynn/lr_scheduler.py`
  - `StepLR`、`MultiStepLR`、`ExponentialLR`
- `run_experiments.py`
  - 训练 MLP、CNN-SGD、CNN-Momentum 三组实验
  - 在 `../../submission` 下保存模型、指标和报告图表

## 运行方式

从项目根目录运行：

```powershell
$env:PYTHONPATH='C:\Users\邓凯源\Desktop\project1\PJ1\codes'
& "C:\Users\邓凯源\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 .\PJ1\codes\sanity_checks.py
& "C:\Users\邓凯源\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 .\PJ1\codes\test_train.py
```

快速流程测试：

```powershell
$env:PYTHONPATH='C:\Users\邓凯源\Desktop\project1\PJ1\codes'
& "C:\Users\邓凯源\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 .\PJ1\codes\test_train.py --quick
```

评估已保存模型：

```powershell
$env:PYTHONPATH='C:\Users\邓凯源\Desktop\project1\PJ1\codes'
& "C:\Users\邓凯源\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 .\PJ1\codes\test_model.py --model .\submission\checkpoints\cnn_momentum\best_model.pickle
```

上传 Github 时不要包含 MNIST 数据集、模型权重、生成图表或 `__pycache__` 文件夹。
