"""
Section 10, Option A: load the base model and attach the trained adapter.

    python inference.py --model Qwen/Qwen2.5-1.5B-Instruct \
                        --adapter runs/smoke/final --no-4bit \
                        --prompt "Explain LoRA in two sentences."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default="qwen2.5-7b-dolly-lora/final")
    p.add_argument("--prompt", default="Explain LoRA in two sentences.")
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    base_kwargs = dict(device_map="auto")
    if args.use_4bit:
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        base_kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(args.model, **base_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model)
    # Left padding is correct for generation.
    tok.padding_side = "left"
    messages = [{"role": "user", "content": args.prompt}]
    # return_dict=True gives {input_ids, attention_mask}. On transformers >=5
    # apply_chat_template returns a BatchEncoding rather than a bare tensor, so
    # we pass it through with **inputs (works on older versions too).
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=True, temperature=0.7,
        )
    print(tok.decode(out[0][input_len:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
