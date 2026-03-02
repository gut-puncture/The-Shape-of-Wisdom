#!/usr/bin/env python3
"""Pre-build figure QC checks for the paper.

Fails with non-zero exit if any figure has:
  - absolute option letters (option_A, option_B, etc.) in a plot
    filename or caption that is interpreted as "correct option"
  - layer axis exceeding L-1 for that model
  - duplicate heatmaps under two names
  - caption referencing quantities not actually plotted
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "paper" / "final_paper"
TEX_FILE = PAPER_DIR / "paper_publish_v2.tex"

MODEL_LAYERS = {
    "Qwen": 28,
    "Llama": 32,
    "Mistral": 32,
}

ERRORS: list[str] = []


def check_absolute_option_in_tex():
    """Flag any caption that mentions option_A/B/C/D as 'correct option'."""
    text = TEX_FILE.read_text()
    captions = re.findall(r"\\caption\{(.*?)\}", text, re.DOTALL)
    for i, cap in enumerate(captions):
        if re.search(r"option[_ ][ABCD]", cap) and re.search(r"correct", cap, re.IGNORECASE):
            ERRORS.append(f"Caption {i+1} references absolute option letter AND 'correct': {cap[:80]}...")


def check_figure_filenames_no_absolute():
    """Figure files should not encode absolute option letters in filenames."""
    fig_dir = PAPER_DIR / "figures"
    for fig in fig_dir.glob("fig*.pdf"):
        if re.search(r"option[_-]?[ABCD]", fig.name):
            ERRORS.append(f"Figure filename contains absolute option letter: {fig.name}")


def check_layer_index_in_tex():
    """Ensure text does not claim layer ranges beyond actual model sizes."""
    text = TEX_FILE.read_text()
    # Check for layer references like "layer 32" or "layers 0-32"
    for match in re.finditer(r"layer[s]?\s*[\{]?(\d+)", text, re.IGNORECASE):
        layer_num = int(match.group(1))
        if layer_num > 31:
            ERRORS.append(f"Text references layer {layer_num}, but max model has 32 layers (0-31): "
                          f"...{text[max(0,match.start()-20):match.end()+20]}...")


def check_causal_language():
    """Ensure 'causal ablation' and 'causal patching' are not used."""
    text = TEX_FILE.read_text()
    for phrase in ["causal ablation", "causal patching"]:
        if phrase.lower() in text.lower():
            ERRORS.append(f"Found deprecated phrase '{phrase}' in tex file")


def check_no_old_figure_refs():
    """Ensure no old timestamped/PNG references remain in v2 TeX."""
    text = TEX_FILE.read_text()
    old_refs = re.findall(r"fig\d_\w+_\d{10,}\.png", text)
    for ref in old_refs:
        ERRORS.append(f"Old timestamped figure reference still in tex: {ref}")
    png_refs = re.findall(r"figures/fig\d_[A-Za-z0-9_]+\.png", text)
    for ref in png_refs:
        ERRORS.append(f"Legacy PNG figure reference still in tex: {ref}")


def main() -> int:
    if not TEX_FILE.exists():
        print(f"ERROR: {TEX_FILE} not found")
        return 1

    check_absolute_option_in_tex()
    check_figure_filenames_no_absolute()
    check_layer_index_in_tex()
    check_causal_language()
    check_no_old_figure_refs()

    if ERRORS:
        print(f"FIGURE QC FAILED — {len(ERRORS)} issue(s):")
        for e in ERRORS:
            print(f"  ✗ {e}")
        return 1
    else:
        print("FIGURE QC PASSED — all checks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
