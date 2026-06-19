# Fine-Tuning an LLM with LoRA and QLoRA: A Hands-On Guide

Full fine-tuning of a modern language model means updating every weight. For a 7B model in 16-bit precision, the optimizer states alone (Adam keeps two extra tensors per parameter) push the memory requirement above 80 GB. Most people do not have that hardware.

LoRA (Low-Rank Adaptation) [1] removes this problem. You freeze the base model and train only a small pair of low-rank matrices that are added next to the original weights. QLoRA [2] goes one step further: it loads the frozen base model in 4-bit precision, so a 7B model fits on a single 16 GB GPU and a 70B model fits on a single 48 GB GPU.

This guide is code-first. The theory is kept to the minimum you need to set the hyperparameters correctly. By the end you will have a working training script that you can run on a single consumer GPU (or a free Colab instance with a smaller model) and an adapter that you can load for inference or merge back into the base model for deployment.

What you need: one GPU with at least 16 GB of VRAM, Python 3.10+, and a Hugging Face account if you use a gated model. The example model here (`Qwen/Qwen2.5-7B-Instruct`) is not gated, so you can start without any access request.

> **Runnable code.** Every snippet below is assembled into three scripts you can
> run directly — `train.py` (Sections 3–9), `inference.py` (Section 10A), and
> `merge.py` (Section 10B) — with a `README.md` and `requirements.txt`. If you
> just want to see the pipeline work end to end on a small model in a few
> minutes, run the smoke test from the README before reading on.

---

## 1. The idea in one screen

A linear layer applies `h = W0 · x`, where `W0` is a frozen weight matrix of shape `(d, k)`. LoRA keeps `W0` untouched and adds a low-rank update next to it:

```
h = W0 · x + (alpha / r) · B · A · x
```

Here `A` has shape `(r, k)`, `B` has shape `(d, r)`, and the rank `r` is small (typically 8 to 64). Only `A` and `B` are trained. `A` is initialised from a random Gaussian and `B` is initialised to zero, so at the start the update is exactly zero and training begins from the base model's behaviour.

The number of trainable parameters per layer drops from `d · k` to `r · (d + k)`. For a 4096×4096 projection with `r = 16`, that is about 131K trainable parameters instead of 16.7M, roughly a 128× reduction.

Three terms decide everything:

- **`r` (rank)** — the capacity of the adapter. Higher `r` can fit more, but also costs more memory and can overfit on small datasets.
- **`alpha` (scaling)** — the update is scaled by `alpha / r`. A common convention is `alpha = 2 · r`, which keeps the effective scale stable when you change the rank. (See the rsLoRA note in Section 11 for why the exact scaling matters.)
- **`target_modules`** — which linear layers receive an adapter. Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are the classic choice; adding the MLP projections (`gate_proj`, `up_proj`, `down_proj`) usually improves quality.

QLoRA adds one component on top of this: the frozen `W0` is stored in 4-bit (the NF4 data type), while the LoRA matrices stay in 16-bit. Gradients flow through the 4-bit weights but the weights themselves are never updated, so the precision loss does not accumulate.

`[IMAGE 1 — place here]`
> **Image 1.** The LoRA reparameterisation: the frozen `W` on the left, the trainable rank-`r` matrices `A` and `B` on the right, summed at the output. Source: Figure 1 of the original LoRA paper, Hu et al., 2021 — https://arxiv.org/abs/2106.09685

---

## 2. Environment

The library APIs in this ecosystem change often, so install recent versions and be aware that the three version-sensitive points (`SFTConfig`, `processing_class`, and `max_length` vs `max_seq_length`) are noted inline below.

```bash
pip install -U "transformers>=4.46" "trl>=0.12" "peft>=0.13" \
               "datasets>=3.0" "accelerate>=1.0" "bitsandbytes>=0.44" \
               tensorboard
```

`tensorboard` is needed because the training config below sets
`report_to="tensorboard"`; drop it from the list only if you also change that to
`report_to="none"`.

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Free Colab (16 GB T4) alternative: "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## 3. Load the base model in 4-bit

This is the "Q" in QLoRA. `BitsAndBytesConfig` controls the quantization. The four settings below are the standard QLoRA recipe from the paper [2].

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # store frozen weights in 4-bit
    bnb_4bit_quant_type="nf4",              # NF4: the 4-bit type designed for normally-distributed weights
    bnb_4bit_use_double_quant=True,         # quantize the quantization constants too (extra ~0.4 GB saved)
    bnb_4bit_compute_dtype=torch.bfloat16,  # matmuls run in bf16, not 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",                      # place layers on the available GPU(s)
    attn_implementation="flash_attention_2" # drop this line if flash-attn is not installed
)
model.config.use_cache = False              # required: caching is incompatible with gradient checkpointing
```

A note on `bnb_4bit_compute_dtype`: the weights are *stored* in 4-bit, but every matrix multiplication temporarily dequantizes them to `bfloat16`. This is why 4-bit training does not destroy quality. Use `bfloat16` on Ampere or newer GPUs (RTX 30xx/40xx, A100, H100). On older cards (e.g. T4), use `torch.float16` instead.

---

## 4. Tokenizer and chat template

The tokenizer must match the model. For instruction models you also need the chat template, which formats a list of messages into the exact string the model was trained on (with the right special tokens). Never hand-write this format. Always use `apply_chat_template`.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Causal LMs need a pad token. If the model has none, reuse the eos token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # right padding is correct for training
```

---

## 5. Prepare the dataset

The real skill in fine-tuning is data formatting, not the training call. Your dataset must be turned into the model's chat format. I use Databricks Dolly here because it is small, clean, and has a clear instruction/response structure.

```python
raw = load_dataset("databricks/databricks-dolly-15k", split="train")

def to_chat(example):
    user = example["instruction"]
    if example["context"]:
        user += "\n\n" + example["context"]

    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["response"]},
    ]
    # tokenize=False returns the formatted string; SFTTrainer tokenizes it later.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = raw.map(to_chat, remove_columns=raw.column_names)

# Inspect one example. Confirm the special tokens look right before training.
print(dataset[0]["text"])
```

The output of that `print` is the single most important thing to check before you launch a run. If the special tokens (`<|im_start|>`, `<|im_end|>` for Qwen) are missing or duplicated, the model will learn the wrong format and behave strangely at inference time.

`[IMAGE 2 — place here]`
> **Image 2.** A screenshot of your own `print(dataset[0]["text"])` output, showing the formatted training sample with chat special tokens. Source: your own terminal / notebook output. (This is the best "sanity check" image for readers to reproduce.)

---

## 6. The LoRA configuration

This is the heart of the method. Every argument below has a direct effect, so I explain each one inline rather than in prose.

```python
lora_config = LoraConfig(
    r=16,                       # rank: capacity of the adapter (8-64 typical)
    lora_alpha=32,             # scaling; alpha = 2*r is a safe default
    lora_dropout=0.05,         # dropout on the LoRA input; regularizes small datasets
    bias="none",               # do not train bias terms (standard for LoRA)
    task_type="CAUSAL_LM",     # tells PEFT how to wrap the model
    target_modules=[           # which linear layers get an adapter
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    # target_modules="all-linear",  # shortcut: adapt every linear layer
)
```

Two practical shortcuts:

- `target_modules="all-linear"` attaches an adapter to every linear layer automatically. It is the simplest choice and usually a strong baseline.
- For common architectures (Llama, Qwen, Gemma, Mistral), PEFT already knows sensible default targets, so you can omit `target_modules` entirely and it will adapt the attention projections.

You do **not** need to call `prepare_model_for_kbit_training` or `get_peft_model` manually. When you pass `peft_config` to `SFTTrainer` (next step), TRL applies both for you, including the gradient-checkpointing fix required for quantized training.

---

## 7. Training configuration and trainer

`SFTConfig` is a subclass of the standard `TrainingArguments`, so every argument from `TrainingArguments` is available here too. The settings below are tuned for a single 24 GB GPU.

```python
sft_config = SFTConfig(
    output_dir="qwen2.5-7b-dolly-lora",
    num_train_epochs=1,                 # 1-3 epochs is normal for LoRA SFT
    per_device_train_batch_size=2,      # raise until you hit the VRAM ceiling
    gradient_accumulation_steps=8,      # effective batch size = 2 * 8 = 16
    gradient_checkpointing=True,        # trades compute for a large VRAM saving
    learning_rate=2e-4,                 # LoRA tolerates a higher LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                          # use fp16=True on older GPUs
    max_length=1024,                    # newer TRL. Older versions: max_seq_length=1024
    packing=True,                       # pack short samples together; big throughput win
    dataset_text_field="text",          # the column produced in Section 5
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,            # TRL wraps the model with LoRA here
    processing_class=tokenizer,         # newer TRL. Older versions: tokenizer=tokenizer
)
```

The two version-sensitive arguments are flagged in the comments. If you get a `TypeError` about an unexpected keyword, your installed version uses the older name (`max_seq_length`, `tokenizer`); switch to it. (In recent TRL the `tokenizer` argument was removed entirely rather than renamed, so on the newest versions only `processing_class` works.)

One caveat about `packing=True`: it concatenates several short samples into one sequence for throughput, but the samples must not attend to each other across their boundaries. Current TRL only guarantees that isolation with a Flash Attention backend (`attn_implementation="flash_attention_2"` in Section 3). With the default attention implementation, recent TRL versions print a warning that packed samples can cross-contaminate. If you cannot install flash-attn, set `packing=False` for a correct (if slightly slower) run.

A useful habit before the first full run is to confirm how few parameters you are actually training:

```python
trainer.model.print_trainable_parameters()
# Example output:
# trainable params: 40,370,176 || all params: 7,656,000,000 || trainable%: 0.53
```

Around 0.5% trainable is exactly what you want. If this number is suspiciously high, your `target_modules` or `task_type` is probably wrong.

---

## 8. Train

```python
trainer.train()
```

While it runs, watch two things:

- **Training loss** should fall and then flatten. If it is flat from step 0, your learning rate is too low or the data formatting is broken. If it explodes to `NaN`, lower the learning rate or switch `bf16`/`fp16` to match your hardware.
- **GPU memory** (`nvidia-smi`). If you hit out-of-memory, the cheapest fixes in order are: lower `per_device_train_batch_size`, lower `max_length`, then raise `gradient_accumulation_steps` to keep the effective batch size the same.

`[IMAGE 3 — place here]`
> **Image 3.** Your training loss curve from TensorBoard (run `tensorboard --logdir qwen2.5-7b-dolly-lora`). A healthy curve drops quickly in the first few hundred steps and then flattens. Source: your own TensorBoard, or Weights & Biases if you set `report_to="wandb"`.

---

## 9. Save the adapter

LoRA's best property is that the trained artifact is tiny. You are saving only `A` and `B`, not the 7B base model, so the adapter is usually 50-200 MB.

```python
trainer.save_model("qwen2.5-7b-dolly-lora/final")
# Saves adapter_config.json + adapter_model.safetensors only.
```

The saved `adapter_config.json` records `base_model_name_or_path`, so anyone loading the adapter later automatically pulls the correct base model.

---

## 10. Inference

There are two ways to run the fine-tuned model, and the right choice depends on your goal.

### Option A: keep the adapter separate (flexible)

Load the base model, then attach the adapter on top. This keeps the base model untouched, so you can hot-swap different adapters for different tasks without reloading the 7B weights.

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,   # the same 4-bit config as training
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "qwen2.5-7b-dolly-lora/final")
model.eval()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"   # left padding is correct for generation
messages = [{"role": "user", "content": "Explain LoRA in two sentences."}]

# return_dict=True yields {"input_ids", "attention_mask"}. On transformers >= 5
# apply_chat_template returns a BatchEncoding rather than a bare tensor, so we
# unpack it with **inputs. This form also works on older versions.
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to(model.device)
input_len = inputs["input_ids"].shape[1]

out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
print(tok.decode(out[0][input_len:], skip_special_tokens=True))
```

A version note: older Transformers returned a plain token-ID tensor from
`apply_chat_template(..., return_tensors="pt")`, so you may see tutorials that
pass it straight into `model.generate(inputs)` and slice with `inputs.shape[1]`.
Transformers 5 changed this to a `BatchEncoding` (which also carries the
attention mask), so prefer the `return_dict=True` form above.

### Option B: merge into the base model (fast deployment)

For production, merge the adapter into the weights and save a single standalone model. After merging there is no adapter overhead at inference, and the result is a normal Transformers model that vLLM, TGI, or Ollama can serve directly.

```python
from peft import PeftModel

# Merge must be done in 16-bit, NOT on the 4-bit model.
base_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
merged = PeftModel.from_pretrained(base_fp16, "qwen2.5-7b-dolly-lora/final")
merged = merged.merge_and_unload()           # folds B·A into W0
merged.save_pretrained("qwen2.5-7b-dolly-merged")
tokenizer.save_pretrained("qwen2.5-7b-dolly-merged")
```

The important detail people miss: merge into the **16-bit** base model, not the 4-bit one. Merging a LoRA adapter into a 4-bit base and re-quantizing introduces avoidable error. Load the base in `bfloat16` for the merge, as shown above.

(On Transformers 5 the `torch_dtype` argument prints a deprecation warning in favour of `dtype`; both still work. Use `dtype=torch.bfloat16` if you are pinned to a recent version and want to silence it, or keep `torch_dtype` for compatibility with `transformers>=4.46`.)

---

## 11. Choosing hyperparameters

A practical starting table. Move from the defaults only when you have a reason.

| Setting | Default | When to change |
|---|---|---|
| `r` | 16 | Raise to 32-64 for hard tasks or large datasets; lower to 8 for tiny datasets |
| `lora_alpha` | `2 * r` | Keep the ratio; this convention keeps the effective scale stable across ranks |
| `lora_dropout` | 0.05 | Raise to 0.1 if you see overfitting; set to 0 for very large datasets |
| `target_modules` | attention + MLP | Use `"all-linear"` if unsure; attention-only is lighter but weaker |
| `learning_rate` | 2e-4 | 1e-4 to 3e-4 is the usual range; lower if loss is unstable |
| `num_train_epochs` | 1-3 | More epochs overfit fast with LoRA; watch eval loss |

The single most common mistake is over-training. LoRA adapts quickly, and a small instruction dataset run for many epochs will memorise the data and lose general ability. Start with one epoch.

---

## 12. Variants worth knowing

All of these are one-line changes to the configuration you already have.

**QLoRA** [2] — already used above. The 4-bit base via `BitsAndBytesConfig` is what makes this QLoRA rather than plain LoRA.

**DoRA** [6] — decomposes each weight into a magnitude and a direction, and applies LoRA only to the direction. It closes part of the accuracy gap to full fine-tuning, at a small extra cost. Enable it with one flag:

```python
lora_config = LoraConfig(r=16, lora_alpha=32, use_dora=True, ...)
```

`[IMAGE 4 — place here]`
> **Image 4.** The DoRA decomposition of a weight into magnitude `m` and direction `V`, with LoRA applied to the directional component. Source: Figure 1 of the DoRA paper, Liu et al., 2024 — https://arxiv.org/abs/2402.09353

**rsLoRA** [7] — rank-stabilized LoRA. It changes the scaling from `alpha / r` to `alpha / sqrt(r)`, which keeps gradients stable at high ranks. If you want to use `r = 64` or higher, turn it on:

```python
lora_config = LoraConfig(r=64, lora_alpha=16, use_rslora=True, ...)
```

**LoRA+** [8] — uses a higher learning rate for the `B` matrix than for `A`, which speeds up convergence. It is available through the PEFT `LoraPlus` optimizer utility rather than a `LoraConfig` flag.

---

## 13. Common pitfalls

- **`use_cache=True` with gradient checkpointing.** You must set `model.config.use_cache = False` during training, then set it back to `True` for inference.
- **Wrong `compute_dtype` for the GPU.** `bfloat16` needs Ampere or newer. On a T4 use `float16`, or training silently misbehaves.
- **Merging into a 4-bit model.** Always merge into the 16-bit base (Section 10, Option B).
- **Padding side.** Use right padding for training and left padding for generation. Mixing them up corrupts the attention mask.
- **Forgetting the chat template.** Training on raw text while the model expects a chat format is the most frequent cause of "it trained fine but the output is nonsense."
- **Too many epochs.** LoRA overfits faster than full fine-tuning. Validate, do not just train.

---

## 14. Evaluating the result

Loss going down is necessary but not sufficient. Hold out a validation split and watch its loss for overfitting (pass `eval_dataset` and set `eval_strategy="steps"` in `SFTConfig`). For instruction tuning, the loss number alone is a weak signal, so also read a handful of generated outputs by hand on prompts the model did not see in training. For tasks with a correct answer (classification, extraction, code), measure task accuracy on a held-out set, because that is the number that actually matters.

---

## References

[1] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. https://arxiv.org/abs/2106.09685

[2] T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023. https://arxiv.org/abs/2305.14314

[3] Hugging Face, "PEFT documentation." https://huggingface.co/docs/peft

[4] Hugging Face, "TRL — Supervised Fine-tuning Trainer." https://huggingface.co/docs/trl/sft_trainer

[5] bitsandbytes-foundation, "bitsandbytes." https://github.com/bitsandbytes-foundation/bitsandbytes

[6] S.-Y. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," 2024. https://arxiv.org/abs/2402.09353

[7] D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)," 2023. https://arxiv.org/abs/2312.03732

[8] S. Hayou, N. Ghosh, B. Yu, "LoRA+: Efficient Low Rank Adaptation of Large Models," 2024. https://arxiv.org/abs/2402.12354
