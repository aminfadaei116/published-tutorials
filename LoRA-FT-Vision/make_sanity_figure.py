"""Reproducible 'sanity-check' figure for Section 5 of the VLM LoRA guide.

Loads ChartQA's first example with the exact to_chat() mapping from train.py,
captures the print(...) output, and renders it beside the actual chart image.
"""
import io
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

# ---- Section 5: prepare the dataset (same mapping as train.py) ----
raw = load_dataset("HuggingFaceM4/ChartQA", split="train[:1]")


def to_chat(example):
    answer = example["label"]
    if isinstance(answer, list):
        answer = answer[0]
    return {
        "images": [example["image"]],
        "prompt": [{"role": "user", "content": example["query"]}],
        "completion": [{"role": "assistant", "content": str(answer)}],
    }


dataset = raw.map(to_chat, remove_columns=raw.column_names)

# Capture the exact sanity-check print block from train.py.
buf = io.StringIO()
print("=== Sanity check: dataset[0] ===", file=buf)
print("prompt:    ", dataset[0]["prompt"], file=buf)
print("completion:", dataset[0]["completion"], file=buf)
print("image size:", dataset[0]["images"][0].size, file=buf)
print("=== end sample ===", file=buf)
printed = buf.getvalue()
print(printed)  # also to real stdout

# Wrap long lines so the console text fits in the figure panel.
wrapped = "\n".join(
    textwrap.fill(line, width=64, subsequent_indent="    ")
    for line in printed.splitlines()
)

img = dataset[0]["images"][0]

fig, (ax_img, ax_txt) = plt.subplots(
    1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1]}
)

ax_img.imshow(img)
ax_img.set_title(f"dataset[0] image  ({img.size[0]}×{img.size[1]})", fontsize=11)
ax_img.axis("off")

ax_txt.axis("off")
ax_txt.set_title("print(...) output  (train.py, Section 5)", fontsize=11)
ax_txt.text(
    0.0, 0.98, wrapped, family="monospace", fontsize=10,
    va="top", ha="left", transform=ax_txt.transAxes,
    bbox=dict(boxstyle="round,pad=0.6", fc="#f5f5f5", ec="#cccccc"),
)

fig.suptitle("Section 5 sanity check: ChartQA prompt/completion + rendered chart",
             fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "assets/sanity_check.png"
import os
os.makedirs("assets", exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nSaved figure to {out}")
