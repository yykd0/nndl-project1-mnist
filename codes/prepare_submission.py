import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission"
CODE_PACKAGE_DIR = SUBMISSION_DIR / "code_for_github"
CHECKPOINT_DIR = SUBMISSION_DIR / "checkpoints"
FIGURE_DIR = SUBMISSION_DIR / "figures"


EXCLUDED_DIRS = {
    "__pycache__",
    "dataset",
    "best_models",
    "saved_models",
    "figs",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pickle", ".pkl", ".gz", ".zip", ".ipynb"}


def should_copy(path):
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def copy_code():
    target = CODE_PACKAGE_DIR / "codes"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for source in BASE_DIR.rglob("*"):
        if source.is_dir() or not should_copy(source.relative_to(BASE_DIR)):
            continue
        relative = source.relative_to(BASE_DIR)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    archive_base = SUBMISSION_DIR / "code_for_github"
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=CODE_PACKAGE_DIR)
    return Path(zip_path)


def write_submission_readme(zip_path):
    readme = SUBMISSION_DIR / "README_submission.md"
    text = f"""# Project 1 提交材料说明

## 文件清单

- `PJ1_report.docx`：中文 Word 报告。Github 链接已填入；正式提交前请替换姓名、学号和模型权重链接占位。
- `code_for_github/`：用于上传 Github 的源码目录，已排除 MNIST 数据、模型权重、生成图表、`__pycache__` 和其他大文件。
- `code_for_github.zip`：源码目录的压缩包版本。
- `checkpoints/`：本地训练得到的模型权重。请上传到 ModelScope 或其他文件托管平台，再把链接填入报告。
- `figures/`：Word 报告中使用的图表源文件。
- `experiment_results.json`：报告使用的实验指标。

## 建议提交步骤

1. 将 `code_for_github/` 的内容或 `{zip_path.name}` 上传到 Github。
2. 将 `checkpoints/mlp_sgd/best_model.pickle`、`checkpoints/cnn_sgd/best_model.pickle` 和 `checkpoints/cnn_momentum/best_model.pickle` 上传到模型权重托管平台。
3. 打开 `PJ1_report.docx`，替换所有占位信息，并按个人需要微调；如果 eLearning 要求 PDF，再由 Word 导出最终 PDF。

## 复现命令

在 `C:\\Users\\邓凯源\\Desktop\\project1` 下运行：

```powershell
$env:PYTHONPATH='C:\\Users\\邓凯源\\Desktop\\project1\\PJ1\\codes'
& "C:\\Users\\邓凯源\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" -X utf8 .\\PJ1\\codes\\sanity_checks.py
& "C:\\Users\\邓凯源\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" -X utf8 .\\PJ1\\codes\\test_train.py
& "C:\\Users\\邓凯源\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" -X utf8 .\\PJ1\\codes\\make_report.py
& "C:\\Users\\邓凯源\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" -X utf8 .\\PJ1\\codes\\prepare_submission.py
```
"""
    readme.write_text(text, encoding="utf-8-sig")
    return readme


def main():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    CODE_PACKAGE_DIR.mkdir(exist_ok=True)
    zip_path = copy_code()
    readme = write_submission_readme(zip_path)
    print(f"code_package={CODE_PACKAGE_DIR}")
    print(f"zip={zip_path}")
    print(f"readme={readme}")


if __name__ == "__main__":
    main()
