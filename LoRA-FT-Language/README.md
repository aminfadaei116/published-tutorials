# LoRA-FT — runnable companion to the LoRA/QLoRA guide

Runnable code for the blog post
[`finetuning-lora-qlora-guide.md`](finetuning-lora-qlora-guide.md). Train a LoRA
(or QLoRA) adapter on an instruction dataset, run inference with it, and merge
it back into the base model.

Every script maps section-by-section to the post, so you can read the post with
one of these files open next to it.

| File | Post section | What it does |
|---|---|---|
| [`train.py`](train.py) | 3–9 | Load base model (4-bit), format data, train the adapter, save it |
| [`inference.py`](inference.py) | 10A | Load base + adapter and generate |
| [`merge.py`](merge.py) | 10B | Merge the adapter into the 16-bit base for deployment |

## Setup

```bash
conda create -y -n LoRA_FT python=3.11
conda activate LoRA_FT
pip install -r requirements.txt
```

A GPU is required for the 4-bit (QLoRA) path. For the example model you want
~16 GB of VRAM; the smoke test below runs on much less.

## Quick smoke test (a few minutes)

Verifies the whole pipeline on a small model with a tiny data subset and only a
handful of steps. Trains in bf16 (`--no-4bit`) so it works even without
bitsandbytes 4-bit support:

```bash
# 1. train a tiny adapter
python train.py --model Qwen/Qwen2.5-1.5B-Instruct \
                --max-steps 10 --subset-size 256 --no-4bit \
                --output-dir runs/smoke

# 2. generate with it
python inference.py --model Qwen/Qwen2.5-1.5B-Instruct \
                    --adapter runs/smoke/final --no-4bit \
                    --prompt "Explain LoRA in two sentences."

# 3. merge it
python merge.py --model Qwen/Qwen2.5-1.5B-Instruct \
                --adapter runs/smoke/final --out runs/smoke-merged
```

## Full run (the recipe from the post)

```bash
# Qwen2.5-7B in 4-bit, Dolly-15k, 1 epoch — needs ~16-24 GB VRAM.
python train.py
python inference.py --adapter qwen2.5-7b-dolly-lora/final
```

Add `--attn flash_attention_2` if you have flash-attn installed. On older GPUs
(e.g. T4) add `--fp16`, which switches both the compute dtype and the trainer to
fp16.

## Useful flags

`train.py`: `--model --dataset --no-4bit --attn --r --alpha --dropout --epochs
--batch-size --grad-accum --lr --max-length --fp16 --max-steps --subset-size`

Run `python train.py --help` for the full list.

## Tested with

Python 3.11, torch 2.12 (CUDA 13), transformers 5.12, trl 1.6, peft 0.19,
datasets 5.0, accelerate 1.14, bitsandbytes 0.49, on an NVIDIA H200. The minimum
versions in `requirements.txt` are the ones the post targets.
