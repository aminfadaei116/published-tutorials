# Blog: Audio & Music Understanding LLMs

This folder contains a technical blog post comparing five open audio-language
models on music understanding, plus all the runnable code to reproduce it.

## Contents

- **[audio-language-models.md](audio-language-models.md)** — the blog post. Start here.
- **[code/](code/)** — one standalone, runnable script per model:
  - [qwen2_audio.py](code/qwen2_audio.py) — Qwen2-Audio-7B-Instruct
  - [audio_flamingo3.py](code/audio_flamingo3.py) — NVIDIA Audio Flamingo 3
  - [gemma4_audio.py](code/gemma4_audio.py) — Google Gemma 4 (E4B)
  - [moss_audio.py](code/moss_audio.py) — OpenMOSS MOSS-Audio-8B
  - [mu_llama.py](code/mu_llama.py) — MU-LLaMA (MERT + LLaMA-7B)
  - [SETUP.md](code/SETUP.md) — conda environments + install steps for each model
- **[results/](results/)** — raw benchmark data: per-model JSON + the
  [full report](results/full_report.md) with every model's description of every clip.

## Quick start

Every script takes `--audio <path>` and one or more `-q "question"` flags (the
model loads once and answers each in turn). See [code/SETUP.md](code/SETUP.md) for
the per-model environment, then e.g.:

```bash
cd code
conda activate persona
python audio_flamingo3.py --audio ../../samples/Grieg_Lyric_Pieces_Kobold.wav \
    -q "What instruments and mood do you hear?"
```
