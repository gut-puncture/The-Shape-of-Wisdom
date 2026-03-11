from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter


def create_qc_booklet(*, figure_pngs: list[Path], table_csvs: list[Path], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for png_path in figure_pngs:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.imshow(mpimg.imread(png_path))
            ax.axis("off")
            ax.set_title(png_path.name, fontsize=11)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
        for csv_path in table_csvs:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            content = csv_path.read_text(encoding="utf-8").splitlines()[:60]
            ax.text(0.01, 0.99, csv_path.name + "\n\n" + "\n".join(content), va="top", ha="left", family="monospace", fontsize=7)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    normalize_pdf(out_pdf, title="qc_booklet")


def rasterize_pdf(pdf_path: Path, output_prefix: Path) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-png", str(pdf_path), str(output_prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    stem = output_prefix.name
    return sorted(output_prefix.parent.glob(f"{stem}-*.png"))


def normalize_pdf(pdf_path: Path, *, title: str) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": "Codex",
            "/Subject": "No-new-inference decision dynamics analysis output",
            "/Creator": "Codex",
            "/Producer": "pypdf",
            "/CreationDate": "D:20000101000000",
            "/ModDate": "D:20000101000000",
        }
    )
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(pdf_path.parent)) as handle:
        writer.write(handle)
        tmp_name = handle.name
    Path(tmp_name).replace(pdf_path)
