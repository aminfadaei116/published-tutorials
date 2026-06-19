# LoRA-FT-Vision — runnable companion to the VLM LoRA/QLoRA guide

Runnable code for the guide
[`finetuning-vlm-lora-qlora-guide.md`](finetuning-vlm-lora-qlora-guide.md).
Fine-tune a **vision-language model** (image + text → text) with LoRA / QLoRA,
run inference on an image, and merge the adapter back into the base model.

This is the vision counterpart to the text-only `LoRA-FT` repo. The method is
the same; the moving parts that differ are: you load an image-text model, you
use a **processor** instead of a tokenizer, your dataset carries images, and a
few TRL settings are VLM-specific (`packing=False`, prompt/completion loss).

| File | Guide section | What it does |
|---|---|---|
| [`train.py`](train.py) | 3–9 | Load VLM (4-bit), format image+QA data, train the adapter, save it |
| [`inference.py`](inference.py) | 10 | Load base + adapter and answer a question about an image |
| [`merge.py`](merge.py) | 11 | Merge the adapter into the 16-bit base for deployment |

## Setup

This reuses the same conda env as the text repo, plus a couple of vision extras:

```bash
conda activate LoRA_FT          # the env from ../LoRA-FT
pip install -r requirements.txt
```

(If you don't already have that env: `conda create -y -n LoRA_FT python=3.11`
first, then the install above.)

A GPU is required. The example model (Qwen2.5-VL-3B) needs ~10-16 GB of VRAM in
4-bit; the smoke test below runs in a few minutes.

## Quick smoke test

Trains a tiny adapter on a ChartQA subset, then uses it to read a chart:

```bash
# 1. train a tiny adapter (4-bit QLoRA, ~7 min)
python train.py --max-steps 40 --subset-size 512 --batch-size 4 \
                --output-dir runs/smoke

# 2. answer a question about an image (local path or URL)
python inference.py --adapter runs/smoke/final \
                    --image /path/to/chart.png \
                    --prompt "How many food items are shown in the bar graph?"

# 3. merge it into a standalone model
python merge.py --adapter runs/smoke/final --out runs/smoke-merged
```

## Full run (the recipe from the guide)

```bash
# Qwen2.5-VL-3B in 4-bit, ChartQA, 1 epoch.
python train.py
python inference.py --adapter qwen2.5-vl-3b-chartqa-lora/final --image your_chart.png
```

Add `--attn flash_attention_2` if you have flash-attn installed. On older GPUs
(e.g. T4) add `--fp16`. Use `--no-4bit` to train in bf16 (plain LoRA). By
default the vision encoder is frozen; pass `--tune-vision` to adapt it too.

## Useful flags

`train.py`: `--model --dataset --no-4bit --attn --max-pixels --min-pixels --r
--alpha --dropout --tune-vision --epochs --batch-size --grad-accum --lr
--max-length --fp16 --max-steps --subset-size`

Run `python train.py --help` for the full list.

## Tested with

Python 3.11, torch 2.12 (CUDA 13), transformers 5.12, trl 1.6, peft 0.19,
datasets 5.0, accelerate 1.14, bitsandbytes 0.49, qwen-vl-utils 0.0.14, on an
NVIDIA H200. Verified end to end: train (loss ~0.2 / token-acc ~0.94 on the
smoke subset) → inference (coherent chart answer) → merge.
