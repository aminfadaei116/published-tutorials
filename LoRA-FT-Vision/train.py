"""
QLoRA / LoRA supervised fine-tuning for a vision-language model (VLM), end to end.

Companion code to the guide
"Fine-Tuning a Vision-Language Model with LoRA and QLoRA: A Hands-On Guide".
Each block maps to a numbered section of the guide.

Quick smoke test (4-bit, tiny subset, a few steps):

    python train.py --max-steps 5 --subset-size 64 --output-dir runs/smoke

Full run from the guide (Qwen2.5-VL-3B in 4-bit on a single 16-24 GB GPU):

    python train.py
"""

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT for a VLM")
    # Section 2 — model / data.
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset", default="HuggingFaceM4/ChartQA")
    p.add_argument("--output-dir", default="qwen2.5-vl-3b-chartqa-lora")

    # Section 3 — quantization.
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false",
                   help="disable 4-bit loading (plain LoRA instead of QLoRA)")
    p.add_argument("--attn", default=None,
                   help="attn implementation, e.g. flash_attention_2 (default: library default)")

    # Section 4 — cap visual tokens. Big images blow up memory because each
    # 28x28 patch becomes a token. max-pixels bounds the longer side.
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28,
                   help="max pixels per image fed to the vision encoder")
    p.add_argument("--min-pixels", type=int, default=4 * 28 * 28)

    # Section 6 — LoRA.
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--tune-vision", action="store_true",
                   help="also attach adapters to the vision tower (default: freeze it)")

    # Section 7 — training.
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--fp16", action="store_true",
                   help="use fp16 instead of bf16 (older GPUs, e.g. T4)")

    # Knobs for a fast smoke test.
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--subset-size", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    # ---- Section 3: load the base VLM (in 4-bit for QLoRA) ---------------
    # Only the language model is quantized; the vision encoder is small and
    # stays in compute dtype.
    model_kwargs = dict(device_map="auto", dtype=compute_dtype)
    if args.attn:
        model_kwargs["attn_implementation"] = args.attn
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            # Keep the vision tower out of 4-bit; quantizing it hurts quality.
            llm_int8_skip_modules=["visual", "lm_head"],
        )

    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- Section 4: processor (tokenizer + image processor) -------------
    # A VLM uses a *processor*, not just a tokenizer. min/max_pixels bound how
    # many visual tokens each image becomes.
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )

    # ---- Section 5: prepare the dataset ---------------------------------
    # We use the *prompt-completion* format: {"images", "prompt", "completion"}.
    # This is important for VLMs — TRL then masks the prompt (which contains the
    # hundreds of image-placeholder tokens) and computes the loss ONLY on the
    # assistant's answer. With the alternative "messages" format, loss is taken
    # over every token, including image placeholders, which inflates it badly.
    # Plain-string content is fine; TRL inserts the <image> placeholder.
    split = "train" if args.subset_size < 0 else f"train[:{args.subset_size}]"
    raw = load_dataset(args.dataset, split=split)

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
    print("\n=== Sanity check: dataset[0] ===")
    print("prompt:    ", dataset[0]["prompt"])
    print("completion:", dataset[0]["completion"])
    print("image size:", dataset[0]["images"][0].size)
    print("=== end sample ===\n")

    # ---- Section 6: LoRA config -----------------------------------------
    # Adapt the language model's attention + MLP projections. By default we
    # freeze the vision tower (exclude "visual.*") which is the standard recipe.
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=None if args.tune_vision else r".*visual.*",
    )

    # ---- Section 7: training config and trainer -------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=not args.fp16,
        fp16=args.fp16,
        max_length=args.max_length,
        packing=False,            # REQUIRED for VLMs; TRL raises otherwise
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=processor,   # a processor -> TRL trains in VLM mode
    )

    trainer.model.print_trainable_parameters()

    # ---- Section 8: train -----------------------------------------------
    trainer.train()

    # ---- Section 9: save the adapter ------------------------------------
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nAdapter saved to {final_dir}")


if __name__ == "__main__":
    main()
