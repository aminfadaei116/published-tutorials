# published-tutorials

Runnable companion code for my published tutorials. Each subfolder is a
self-contained mini-project that maps section-by-section to a blog post, so you
can read the post with the code open next to it.

## Tutorials

| Folder | Topic | Post |
|---|---|---|
| [LoRA-FT-Language](LoRA-FT-Language) | Fine-tuning a text LLM with **LoRA / QLoRA** — train an adapter on an instruction dataset, run inference, merge for deployment. | [finetuning-lora-qlora-guide.md](LoRA-FT-Language/finetuning-lora-qlora-guide.md) |
| [LoRA-FT-Vision](LoRA-FT-Vision) | The **vision-language** counterpart — fine-tune an image+text → text model with LoRA / QLoRA. | [finetuning-vlm-lora-qlora-guide.md](LoRA-FT-Vision/finetuning-vlm-lora-qlora-guide.md) |

`LoRA-FT-Vision` is the vision version of `LoRA-FT-Language`: the method is the same; what
differs is loading an image-text model, using a processor instead of a
tokenizer, image-carrying datasets, and a few VLM-specific training settings.

## Layout

Each folder follows the same shape:

```
<tutorial>/
├── <guide>.md          the blog post itself
├── README.md           setup, smoke test, full run
├── train.py            train the LoRA/QLoRA adapter
├── inference.py        load base + adapter and generate
├── merge.py            merge the adapter into the base for deployment
└── requirements.txt    the stack for that tutorial
```

## Getting started

Pick a folder and follow its `README.md` — it has the conda setup, a quick smoke
test that runs in a few minutes, and the full training recipe. A GPU is required
for the 4-bit (QLoRA) paths.

```bash
cd LoRA-FT-Language   # or LoRA-FT-Vision
cat README.md
```

Training artifacts (adapters, merged models, logs, `*.safetensors`) are
gitignored and not committed.
