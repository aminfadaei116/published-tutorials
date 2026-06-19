"""
Section 11, Option B: merge the adapter into the 16-bit base VLM and save a
standalone model that vLLM / TGI can serve directly.

    python merge.py --adapter runs/smoke/final --out runs/smoke-merged

Note: merge into the 16-bit base, NEVER the 4-bit one.
"""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default="qwen2.5-vl-3b-chartqa-lora/final")
    p.add_argument("--out", default="qwen2.5-vl-3b-chartqa-merged")
    return p.parse_args()


def main():
    args = parse_args()

    base_fp16 = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base_fp16, args.adapter)
    merged = merged.merge_and_unload()  # folds B*A into W0
    merged.save_pretrained(args.out)

    AutoProcessor.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
