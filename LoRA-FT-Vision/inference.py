"""
Section 10, Option A: load the base VLM and attach the trained adapter, then
answer a question about an image.

    python inference.py --adapter runs/smoke/final \
                        --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg \
                        --prompt "What is shown in this image?"

--image accepts a local path or an http(s) URL.
"""

import argparse
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

DEFAULT_IMAGE = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/transformers/tasks/idefics-few-shot.jpg"
)


def load_image(src):
    if src.startswith(("http://", "https://")):
        return Image.open(BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default="qwen2.5-vl-3b-chartqa-lora/final")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()

    base_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    if args.use_4bit:
        base_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"],
        )

    base = AutoModelForImageTextToText.from_pretrained(args.model, **base_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.adapter)
    image = load_image(args.image)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": args.prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    print(processor.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
