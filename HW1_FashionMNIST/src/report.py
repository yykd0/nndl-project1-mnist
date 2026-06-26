from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def build_report(*args, **kwargs):
    """The polished PDF report is provided in Google Drive.

    The local full version used reportlab to compose the PDF from metrics and figures.
    See the repository README for the report download link.
    """
    print("PDF report: https://drive.google.com/file/d/15sGfMvcHesSWj1FsOF-2868lh-2XLBhl/view?usp=drivesdk")


if __name__ == "__main__":
    build_report()
