"""Generate the Chinese Project 2 PDF report from experiment outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

SIMSUN = Path("C:/Windows/Fonts/simsun.ttc")
SIMSUN_BOLD = Path("C:/Windows/Fonts/simsunb.ttf")
TIMES = Path("C:/Windows/Fonts/times.ttf")
TIMES_BOLD = Path("C:/Windows/Fonts/timesbd.ttf")

FONT_CN = "SimSun"
FONT_CN_BOLD = "SimSun"
FONT_EN = "TimesNewRoman"
FONT_EN_BOLD = "TimesNewRoman-Bold"


def register_fonts() -> None:
    """Register Songti for Chinese and Times New Roman for western text."""
    if not SIMSUN.exists() or not TIMES.exists():
        missing = [str(path) for path in (SIMSUN, TIMES) if not path.exists()]
        raise FileNotFoundError("Required fonts are missing: " + ", ".join(missing))

    try:
        pdfmetrics.registerFont(TTFont(FONT_CN, str(SIMSUN), subfontIndex=0))
    except TypeError:
        pdfmetrics.registerFont(TTFont(FONT_CN, str(SIMSUN)))

    pdfmetrics.registerFont(TTFont(FONT_EN, str(TIMES)))
    if TIMES_BOLD.exists():
        pdfmetrics.registerFont(TTFont(FONT_EN_BOLD, str(TIMES_BOLD)))
    else:
        pdfmetrics.registerFont(TTFont(FONT_EN_BOLD, str(TIMES)))


def is_western_char(char: str) -> bool:
    """Return True for ASCII letters, digits, spaces, and western punctuation."""
    if char.isspace():
        return True
    return ord(char) < 128


def mixed_text(text: object, western_font: str = FONT_EN) -> str:
    """Wrap western runs with Times New Roman while leaving Chinese in Songti."""
    raw = str(text)
    if not raw:
        return ""

    pieces: list[tuple[bool, str]] = []
    buffer: list[str] = []
    mode: bool | None = None

    for char in raw:
        current = is_western_char(char)
        if mode is None:
            mode = current
        if current != mode:
            pieces.append((mode, "".join(buffer)))
            buffer = [char]
            mode = current
        else:
            buffer.append(char)

    if buffer:
        pieces.append((bool(mode), "".join(buffer)))

    rendered: list[str] = []
    for western, chunk in pieces:
        escaped = html.escape(chunk)
        if western:
            rendered.append(f'<font name="{western_font}">{escaped}</font>')
        else:
            rendered.append(escaped)
    return "".join(rendered)


def paragraph(text: object, style: ParagraphStyle, western_font: str = FONT_EN) -> Paragraph:
    return Paragraph(mixed_text(text, western_font=western_font), style)


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def optimizer_label(value: str) -> str:
    return {"adam": "Adam", "sgd_momentum": "SGD+动量"}.get(value, value)


def activation_label(value: str) -> str:
    return {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU"}.get(value, value)


def loss_label(value: str) -> str:
    return {"cross_entropy": "交叉熵", "label_smoothing": "Label smoothing"}.get(value, value)


def add_table(story, headers, rows, styles, widths=None) -> None:
    table_data = [
        [paragraph(cell, styles["TableHeader"], western_font=FONT_EN_BOLD) for cell in headers]
    ]
    for row in rows:
        table_data.append([paragraph(cell, styles["TableCell"]) for cell in row])

    table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))


def add_image(story, path: Path, caption: str, styles, max_width: float = 16.2 * cm) -> None:
    if not path.exists():
        return
    img = Image(str(path))
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = max_width
    img.drawHeight = max_width * ratio
    story.append(img)
    story.append(paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 0.35 * cm))


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_EN, 9)
    canvas.drawCentredString(A4[0] / 2.0, 0.75 * cm, str(doc.page))
    canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = FONT_CN

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=FONT_CN_BOLD,
            fontSize=16,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=0.35 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodyText"],
            fontName=FONT_CN,
            fontSize=10,
            leading=14,
            firstLineIndent=0,
            spaceAfter=0.05 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyIndent",
            parent=styles["BodyText"],
            fontName=FONT_CN,
            fontSize=10,
            leading=15,
            firstLineIndent=20,
            alignment=0,
            spaceAfter=0.18 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName=FONT_CN,
            fontSize=8,
            leading=11,
            firstLineIndent=0,
            textColor=colors.HexColor("#4B5563"),
            alignment=TA_CENTER,
            spaceAfter=0.08 * cm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontName=FONT_CN_BOLD,
            fontSize=7.2,
            leading=9,
            firstLineIndent=0,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName=FONT_CN,
            fontSize=7.0,
            leading=9,
            firstLineIndent=0,
            alignment=TA_CENTER,
        )
    )

    styles["Heading1"].fontName = FONT_CN_BOLD
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 18
    styles["Heading1"].spaceBefore = 0.25 * cm
    styles["Heading1"].spaceAfter = 0.15 * cm
    styles["Heading1"].firstLineIndent = 0

    styles["Heading2"].fontName = FONT_CN_BOLD
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading2"].spaceBefore = 0.18 * cm
    styles["Heading2"].spaceAfter = 0.1 * cm
    styles["Heading2"].firstLineIndent = 0

    return styles


def build_report(args: argparse.Namespace) -> Path:
    register_fonts()
    with (RESULTS_DIR / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    output = REPORTS_DIR / "Project2_Report_DengKaiyuan_CN.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
    )
    styles = build_styles()

    best = max(summary["compact_results"], key=lambda row: row["best_test_accuracy"])
    compact_setup = summary["compact_setup"]
    bn_setup = summary["bn_setup"]

    story = [
        paragraph("项目二报告：CIFAR-10 图像分类与 Batch Normalization 分析", styles["ReportTitle"], western_font=FONT_EN_BOLD),
        paragraph(f"姓名：{args.name}", styles["Meta"]),
        paragraph(f"学号：{args.student_id}", styles["Meta"]),
        paragraph(f"代码链接：{args.code_link}", styles["Meta"]),
        paragraph(f"数据集链接：{args.dataset_link}", styles["Meta"]),
        paragraph(f"训练模型权重链接：{args.weights_link}", styles["Meta"]),
        Spacer(1, 0.18 * cm),
    ]

    story.append(paragraph("摘要", styles["Heading1"], western_font=FONT_EN_BOLD))
    story.append(
        paragraph(
            "本报告围绕 CIFAR-10 图像分类任务和 Batch Normalization 的优化作用展开实验。第一部分实现并比较了一个紧凑卷积网络的多个变体，"
            "覆盖二维卷积层、二维池化层、激活函数、全连接层、BatchNorm、Dropout、不同通道数、不同损失函数和不同优化器。"
            "第二部分在课程给定 VGG-A 结构基础上实现 VGG_A_BatchNorm，并从损失曲线包络、梯度预测性和梯度变化幅度三个角度比较有无 BN 的优化表现。"
            f"本地实验中最优紧凑模型为 {best['name']}，测试准确率为 {pct(best['best_test_accuracy'])}，测试错误率为 {pct(best['best_test_error'])}。",
            styles["BodyIndent"],
        )
    )

    story.append(paragraph("实验设置", styles["Heading1"], western_font=FONT_EN_BOLD))
    story.append(
        paragraph(
            f"所有实验固定随机种子为 {summary['seed']}，运行设备为 {summary['device'].upper()}。"
            f"紧凑网络搜索使用 {compact_setup['train_items']} 张训练图像、{compact_setup['test_items']} 张测试图像，"
            f"batch size 为 {compact_setup['batch_size']}，每个模型训练 {compact_setup['epochs']} 轮。"
            f"Batch Normalization 对比实验使用 {bn_setup['train_items']} 张训练图像、{bn_setup['test_items']} 张测试图像，"
            f"每个学习率训练 {bn_setup['epochs']} 轮，学习率集合为 {bn_setup['learning_rates']}。"
            "受本地 CPU 训练时间限制，实验优先保证结构实现、对比流程和可复现结果完整。",
            styles["BodyIndent"],
        )
    )

    story.append(paragraph("一、CIFAR-10 网络训练与优化", styles["Heading1"], western_font=FONT_EN_BOLD))
    story.append(
        paragraph(
            "紧凑分类网络由三组卷积块构成，每组包含两层 2D 卷积、BatchNorm、非线性激活、MaxPool2d 和 Dropout，最后接全连接分类器。"
            "该结构满足课程要求中的全连接层、二维卷积层、二维池化层和激活函数，并额外加入 BatchNorm 与 Dropout。"
            "为优化网络表现，本实验比较了不同基础通道数、不同激活函数、普通交叉熵与 label smoothing、Adam 与 SGD+momentum。",
            styles["BodyIndent"],
        )
    )
    compact_rows = []
    for row in summary["compact_results"]:
        compact_rows.append(
            [
                row["name"].replace("_", " "),
                row["base_channels"],
                activation_label(row["activation"]),
                loss_label(row["loss"]),
                optimizer_label(row["optimizer"]),
                f"{row['parameters']:,}",
                pct(row["best_test_accuracy"]),
                pct(row["best_test_error"]),
            ]
        )
    add_table(
        story,
        ["模型", "通道", "激活", "损失", "优化器", "参数量", "最佳准确率", "测试错误率"],
        compact_rows,
        styles,
        widths=[4.1 * cm, 1.0 * cm, 1.15 * cm, 2.0 * cm, 1.65 * cm, 1.8 * cm, 1.75 * cm, 1.75 * cm],
    )
    story.append(
        paragraph(
            f"最优模型为 {best['name']}，参数量 {best['parameters']:,}。从训练曲线看，GELU 与 SiLU 变体在后期收敛更稳定，"
            "label smoothing 也带来一定泛化收益。第一层卷积核可视化显示，模型学习到了颜色对比和局部边缘等低层特征。",
            styles["BodyIndent"],
        )
    )
    add_image(story, FIGURES_DIR / "compact_cifar_training.png", "图 1：紧凑 CIFAR-10 网络训练曲线。", styles)
    add_image(story, FIGURES_DIR / "compact_first_layer_filters.png", "图 2：最优紧凑模型第一层卷积核可视化。", styles, max_width=9.8 * cm)

    story.append(PageBreak())
    story.append(paragraph("二、Batch Normalization 对优化的影响", styles["Heading1"], western_font=FONT_EN_BOLD))
    story.append(
        paragraph(
            "在 VGG-A 对比实验中，我在每个卷积层后加入 BatchNorm2d，并在分类器中加入 BatchNorm1d，得到 VGG_A_BatchNorm。"
            "实验在相同学习率集合下比较原始 VGG-A 与 BN 版本，记录每一步训练损失，并计算最后线性层梯度的相邻 cosine 相似度和最大梯度差异。"
            "其中损失曲线包络用于观察不同学习率下的损失变化范围，梯度指标用于近似衡量局部线性近似的稳定性。",
            styles["BodyIndent"],
        )
    )
    bn_rows = []
    for row in summary["bn_results"]:
        bn_rows.append(
            [
                "带 BN" if row["model"] == "VGG_A_BatchNorm" else "不带 BN",
                f"{row['learning_rate']:.0e}",
                f"{row['parameters']:,}",
                pct(row["test_accuracy"]),
                f"{row['train_loss']:.3f}",
                f"{row['mean_grad_cosine']:.3f}",
                f"{row['max_grad_diff']:.3f}",
            ]
        )
    add_table(
        story,
        ["模型", "学习率", "参数量", "测试准确率", "训练损失", "梯度相似度", "最大梯度差"],
        bn_rows,
        styles,
        widths=[2.1 * cm, 1.5 * cm, 2.0 * cm, 1.9 * cm, 1.85 * cm, 2.05 * cm, 2.15 * cm],
    )
    add_image(story, FIGURES_DIR / "bn_loss_landscape.png", "图 3：不同学习率下的损失曲线包络。", styles)
    add_image(story, FIGURES_DIR / "bn_gradient_metrics.png", "图 4：有无 BatchNorm 的准确率与梯度统计比较。", styles)

    story.append(PageBreak())
    story.append(paragraph("实验结论", styles["Heading1"], western_font=FONT_EN_BOLD))
    story.append(
        paragraph(
            "在紧凑网络实验中，更合适的激活函数、正则化损失和优化器设置能够明显影响测试误差。"
            "在 VGG-A 对比中，BN 版本的损失包络整体更紧，说明 BN 对训练过程的尺度变化具有缓冲作用，"
            "这与“BN 通过重参数化使优化景观更平滑”的解释一致。由于本地 CPU 训练预算有限，BN 实验更侧重展示趋势与分析流程；"
            "若在 GPU 上使用完整数据集和更多 epoch，绝对准确率仍可继续提升。",
            styles["BodyIndent"],
        )
    )
    story.append(
        paragraph(
            "综上，本项目完成了 CIFAR-10 分类网络构建、结构与优化策略搜索、模型特征可视化、VGG-A BatchNorm 实现、"
            "损失景观可视化和梯度稳定性分析。所有代码、结果表、图表和权重均保存在提交包中，可按 README 中的命令复现实验。",
            styles["BodyIndent"],
        )
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chinese Project 2 report PDF.")
    parser.add_argument("--name", default="邓凯源")
    parser.add_argument("--student-id", default="22300680061")
    parser.add_argument("--code-link", default="https://github.com/yykd0/nndl-project1-mnist/tree/main/project2-cifar-bn")
    parser.add_argument("--dataset-link", default="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    parser.add_argument("--weights-link", default="提交包内 best_compact_cifar.pt")
    return parser.parse_args()


if __name__ == "__main__":
    build_report(parse_args())
