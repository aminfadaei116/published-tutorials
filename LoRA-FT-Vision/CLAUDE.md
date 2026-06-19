# CLAUDE.md — LoRA-FT-Vision

Runnable companion to the guide **"Fine-Tuning a Vision-Language Model with LoRA
and QLoRA: A Hands-On Guide"**
([finetuning-vlm-lora-qlora-guide.md](finetuning-vlm-lora-qlora-guide.md)).
The vision (image + text → text) counterpart to [../LoRA-FT](../LoRA-FT).

## What's here

| File | Purpose |
|---|---|
| [train.py](train.py) | Load a VLM (4-bit by default), format image+QA data, train a LoRA adapter, save it + processor to `<output-dir>/final`. Maps to guide sections 3–9. |
| [inference.py](inference.py) | Load base + adapter and answer a question about an image (local path or URL). Section 10. |
| [merge.py](merge.py) | Merge the adapter into the **16-bit** base for deployment. Section 11. |
| [requirements.txt](requirements.txt) | Adds torchvision, qwen-vl-utils, pillow, requests on top of the text stack. |
| [README.md](README.md) | User-facing setup, smoke test, and full-run instructions. |

## Conventions / things to know

- **Code mirrors the guide**, section by section (`# ---- Section N: ... ----`).
  Keep that mapping intact when editing.
- **Defaults = the guide's recipe.** `Qwen/Qwen2.5-VL-3B-Instruct` on
  `HuggingFaceM4/ChartQA`, 4-bit QLoRA, r=16, alpha=32, 1 epoch. Output dir
  `qwen2.5-vl-3b-chartqa-lora`.

### What differs from the text repo (the VLM-specific parts)

- Uses `AutoModelForImageTextToText` + `AutoProcessor` (a **processor**, not a
  bare tokenizer). Passing a processor as `processing_class` is what puts TRL in
  VLM mode.
- Dataset uses the **prompt/completion** format (`{"images", "prompt",
  "completion"}`) so TRL masks the prompt — including the hundreds of
  image-placeholder tokens — and computes loss only on the answer. Do not switch
  to the `messages` format; it takes loss over image placeholders and inflates it.
- `packing=False` is **required** for VLMs (TRL raises otherwise).
- 4-bit quantization **skips the vision tower** (`llm_int8_skip_modules=
  ["visual", "lm_head"]`); only the language model is quantized.
- Vision tower is **frozen by default** via LoRA `exclude_modules=r".*visual.*"`;
  pass `--tune-vision` to adapt it too.
- `--max-pixels` / `--min-pixels` bound visual tokens (each 28×28 patch → a
  token); large images blow up memory.
- `merge.py` and `inference.py` load the processor from the **adapter** dir
  (it's saved alongside the adapter in `train.py`).

- `--no-4bit` → plain LoRA in bf16; `--fp16` for older GPUs; `--attn
  flash_attention_2` opt-in. Merge targets the 16-bit base, **never** the 4-bit.
- Tested end to end with Python 3.11, torch 2.12 / transformers 5.12 / trl 1.6 /
  peft 0.19 / qwen-vl-utils 0.0.14 on an H200.
