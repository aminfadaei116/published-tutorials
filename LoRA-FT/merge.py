"""
Section 10, Option B: merge the adapter into the 16-bit base model and save a
standalone model that vLLM / TGI / Ollama can serve directly.

    python merge.py --model Qwen/Qwen2.5-1.5B-Instruct \
                    --adapter runs/smoke/final \
                    --out runs/smoke-merged

Note: merge into the 16-bit base, NEVER the 4-bit one.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default="qwen2.5-7b-dolly-lora/final")
    p.add_argument("--out", default="qwen2.5-7b-dolly-merged")
    return p.parse_args()


def main():
    args = parse_args()

    base_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base_fp16, args.adapter)
    merged = merged.merge_and_unload()  # folds B*A into W0
    merged.save_pretrained(args.out)

    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
