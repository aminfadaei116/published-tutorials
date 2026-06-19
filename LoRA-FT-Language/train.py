"""
QLoRA / LoRA supervised fine-tuning, end to end.

This is the runnable companion to the blog post
"Fine-Tuning an LLM with LoRA and QLoRA: A Hands-On Guide".
Each block maps to a numbered section of the post.

Quick smoke test (small model, a few steps, no 4-bit needed):

    python train.py --model Qwen/Qwen2.5-1.5B-Instruct \
                    --max-steps 10 --subset-size 256 --no-4bit \
                    --output-dir runs/smoke

Full run from the post (7B in 4-bit on a 16-24 GB GPU):

    python train.py
"""

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT")
    # Section 2 — model. Default is the post's example; the 1.5B is the
    # free-Colab / quick-test alternative.
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    p.add_argument("--output-dir", default="qwen2.5-7b-dolly-lora")

    # Section 3 — quantization. --no-4bit trains in bf16 (handy for small models
    # or GPUs without bitsandbytes support).
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false",
                   help="disable 4-bit loading (plain LoRA instead of QLoRA)")
    p.add_argument("--attn", default=None,
                   help="attn implementation, e.g. flash_attention_2 (default: library default)")

    # Section 6 — LoRA.
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)

    # Section 7 — training.
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--fp16", action="store_true",
                   help="use fp16 instead of bf16 (older GPUs, e.g. T4)")

    # Knobs for a fast smoke test. Both default to "off".
    p.add_argument("--max-steps", type=int, default=-1,
                   help="cap total optimizer steps (>0 overrides epochs)")
    p.add_argument("--subset-size", type=int, default=-1,
                   help="train on only the first N examples")
    return p.parse_args()


def main():
    args = parse_args()
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16

    # ---- Section 3: load the base model (in 4-bit for QLoRA) -------------
    model_kwargs = dict(device_map="auto")
    if args.attn:
        model_kwargs["attn_implementation"] = args.attn
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- Section 4: tokenizer and chat template -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Section 5: prepare the dataset ---------------------------------
    split = "train" if args.subset_size < 0 else f"train[:{args.subset_size}]"
    raw = load_dataset(args.dataset, split=split)

    def to_chat(example):
        user = example["instruction"]
        if example.get("context"):
            user += "\n\n" + example["context"]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = raw.map(to_chat, remove_columns=raw.column_names)
    print("\n=== Sanity check: dataset[0]['text'] ===")
    print(dataset[0]["text"])
    print("=== end sample ===\n")

    # ---- Section 6: LoRA config -----------------------------------------
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
    )

    # ---- Section 7: training config and trainer -------------------------
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=not args.fp16,
        fp16=args.fp16,
        max_length=args.max_length,
        packing=True,
        dataset_text_field="text",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.model.print_trainable_parameters()

    # ---- Section 8: train -----------------------------------------------
    trainer.train()

    # ---- Section 9: save the adapter ------------------------------------
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    print(f"\nAdapter saved to {final_dir}")


if __name__ == "__main__":
    main()
