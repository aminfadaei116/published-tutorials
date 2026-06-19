# CLAUDE.md — LoRA-FT

Runnable companion to the blog post **"Fine-Tuning an LLM with LoRA and QLoRA: A
Hands-On Guide"** ([finetuning-lora-qlora-guide.md](finetuning-lora-qlora-guide.md)).
Text-only causal LM fine-tuning with LoRA / QLoRA.

## What's here

| File | Purpose |
|---|---|
| [train.py](train.py) | Load base model (4-bit by default), format an instruction dataset to chat text, train a LoRA adapter, save it to `<output-dir>/final`. Maps to post sections 3–9. |
| [inference.py](inference.py) | Load base + adapter (`PeftModel`) and generate. Section 10A. |
| [merge.py](merge.py) | Merge the adapter into the **16-bit** base (`merge_and_unload`) for deployment. Section 10B. |
| [requirements.txt](requirements.txt) | Core stack: torch, transformers, trl, peft, datasets, accelerate, bitsandbytes. |
| [README.md](README.md) | User-facing setup, smoke test, and full-run instructions. |

## Conventions / things to know

- **Code mirrors the post.** Each block in `train.py` is labelled with the post
  section it corresponds to (`# ---- Section N: ... ----`). When editing, keep
  that section-by-section mapping intact — readers follow along with the post
  open. Update both the code and the prose if behavior changes.
- **Defaults = the post's recipe.** `Qwen/Qwen2.5-7B-Instruct` on
  `databricks/databricks-dolly-15k`, 4-bit QLoRA (nf4 + double quant), r=16,
  alpha=32, 1 epoch. Output dir `qwen2.5-7b-dolly-lora`.
- **Smoke-test path** uses `--model Qwen/Qwen2.5-1.5B-Instruct --no-4bit
  --max-steps --subset-size` so the whole pipeline runs in minutes without
  bitsandbytes 4-bit support.
- `--no-4bit` → plain LoRA in bf16. `--fp16` switches compute + trainer to fp16
  (older GPUs like T4). `--attn flash_attention_2` is opt-in.
- Adapter is always saved to `<output-dir>/final`. Merge targets the 16-bit
  base, **never** the 4-bit one.
- LoRA targets the attention + MLP projections (`q/k/v/o_proj`,
  `gate/up/down_proj`); `task_type="CAUSAL_LM"`.
- Tested with Python 3.11, torch 2.12 / transformers 5.12 / trl 1.6 / peft 0.19
  on an H200; `requirements.txt` pins the minimum versions the post targets.

## Sibling

[../LoRA-FT-Vision](../LoRA-FT-Vision) is the vision-language (image+text)
counterpart to this repo — same method, VLM-specific moving parts.
