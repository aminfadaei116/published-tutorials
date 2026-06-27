# Environment setup for the code samples

Each model has conflicting dependencies, so they run in **separate conda
environments**. Below is the minimum to get each script running. All scripts take
`--audio <path>` and one or more `-q "..."` questions (the model loads once and
answers each question in turn). With no `-q`, they fall back to a detailed
"expert music analyst" prompt.

A GPU is strongly recommended. The 7–8B models fit comfortably on a single 24 GB+
card; Audio Flamingo 3 is loaded in fp32 here and is happiest on a larger card.

---

## `persona` env — Qwen2-Audio, Audio Flamingo 3, Gemma 4

These three are all standard HuggingFace `transformers` 5.x models.

```bash
conda create -n persona python=3.12 -y
conda activate persona
pip install "transformers>=5.0" torch librosa soundfile accelerate
# Gemma 4 also needs the vision deps of the unified model + a license + login:
pip install pillow torchvision
huggingface-cli login          # accept the Gemma license on the model page first
```

Run:

```bash
python qwen2_audio.py     --audio ../../samples/Grieg_Lyric_Pieces_Kobold.wav
python audio_flamingo3.py --audio ../../samples/Grieg_Lyric_Pieces_Kobold.wav
python gemma4_audio.py    --audio ../../samples/Grieg_30s.wav   # audio must be <=30s
```

## `moss-audio` env — MOSS-Audio

MOSS-Audio pins older deps and ships custom modeling code, so it needs its own env
and a **local** weight download.

```bash
conda create -n moss-audio python=3.12 -y
conda activate moss-audio
conda install -c conda-forge "ffmpeg=7" -y
git clone https://github.com/OpenMOSS/MOSS-Audio ../../third_party/MOSS-Audio
cd ../../third_party/MOSS-Audio
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
hf download OpenMOSS-Team/MOSS-Audio-8B-Instruct \
    --local-dir ./weights/MOSS-Audio-8B-Instruct
```

Run:

```bash
python moss_audio.py \
    --model-path ../../third_party/MOSS-Audio/weights/MOSS-Audio-8B-Instruct \
    --repo ../../third_party/MOSS-Audio \
    --audio ../../samples/Grieg_Lyric_Pieces_Kobold.wav
```

## `mu-llama` env — MU-LLaMA

```bash
conda create -n mu-llama python=3.10 -y
conda activate mu-llama
git clone https://github.com/crypto-code/MU-LLaMA ../../third_party/MU-LLaMA
cd ../../third_party/MU-LLaMA/MU-LLaMA
pip install -r requirements.txt
# Download LLaMA-7B weights + the MU-LLaMA adapter into mu_ckpts/ (see repo README).
export HF_MODULES_CACHE="$HOME/.cache/hf_modules"   # MERT writes module files here
```

Run:

```bash
python mu_llama.py \
    --repo ../../third_party/MU-LLaMA/MU-LLaMA \
    --audio ../../samples/Grieg_Lyric_Pieces_Kobold.wav
```
