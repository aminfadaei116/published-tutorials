#!/usr/bin/env python
"""
Ask questions about a music audio file with MU-LLaMA (MERT encoder + LLaMA-7B adapter).

MU-LLaMA is NOT a HuggingFace AutoModel. It ships custom code in the
crypto-code/MU-LLaMA GitHub repo: a frozen MERT-v1-330M music encoder feeding a
LLaMA-7B language model through a lightweight adapter. You must clone the repo and
download its checkpoints first.

IMPORTANT — separate environment + setup:
    conda create -n mu-llama python=3.10 -y
    conda activate mu-llama
    git clone https://github.com/crypto-code/MU-LLaMA third_party/MU-LLaMA
    cd third_party/MU-LLaMA/MU-LLaMA
    pip install -r requirements.txt
    # then download the LLaMA-7B weights + the MU-LLaMA adapter checkpoint into
    # mu_ckpts/ (see the repo README). MERT-v1-330M is pulled from the Hub.

    # MERT uses trust_remote_code and writes module files, so point its cache at a
    # writable dir before running:
    export HF_MODULES_CACHE="$HOME/.cache/hf_modules"

Usage (point --repo at the clone; checkpoints default to <repo>/mu_ckpts/):
    python mu_llama.py \
        --repo third_party/MU-LLaMA/MU-LLaMA \
        --audio ../samples/Grieg_Lyric_Pieces_Kobold.wav \
        -q "What instruments do you hear in this music?"
"""
import argparse
import os
import sys

import torch
import torchaudio

# MERT-v1-330M expects 24 kHz mono.
MERT_SR = 24000

# Rich default prompt: ask for a thorough, structured musical breakdown.
DEFAULT_PROMPT = (
    "You are an expert music analyst. Listen to this music very carefully and "
    "give a detailed, structured analysis. Cover each of the following:\n"
    "1. Instrumentation: every instrument you can identify, and the role each plays "
    "(lead, accompaniment, bass, percussion, etc.).\n"
    "2. Genre and subgenre, plus any stylistic influences and the likely era or period.\n"
    "3. Tempo in BPM, the time signature, and the overall rhythmic feel.\n"
    "4. Key and tonality (major/minor), and any notable harmonic or melodic features.\n"
    "5. Dynamics, articulation, and performance techniques.\n"
    "6. Mood, emotion, and tone — and how they evolve over the excerpt.\n"
    "7. Structure and form.\n"
    "8. Production and recording qualities (acoustic vs. electronic, live vs. studio).\n"
    "If there are vocals, describe the voice, delivery, language, and lyrics. "
    "Be specific, descriptive, and confident."
)


def load_audio(path):
    """Mono waveform at MERT's 24 kHz, shaped (1, samples) for the model."""
    wav, sr = torchaudio.load(path)
    if sr != MERT_SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=MERT_SR)
    return torch.stack([torch.mean(wav, 0)], dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="MU-LLaMA music Q&A")
    p.add_argument("--audio", required=True, help="path to an audio file")
    p.add_argument(
        "--repo", default="third_party/MU-LLaMA/MU-LLaMA",
        help="path to the cloned MU-LLaMA repo (provides the `llama` package)",
    )
    p.add_argument("--model-path", default=None, help="adapter checkpoint.pth")
    p.add_argument("--llama-dir", default=None, help="dir with LLaMA-7B weights")
    p.add_argument("--mert-path", default="m-a-p/MERT-v1-330M")
    p.add_argument(
        "-q", "--question", action="append", default=None,
        help="question about the audio (repeatable). Defaults to a detailed prompt.",
    )
    p.add_argument("--max-gen-len", type=int, default=512)
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    return p.parse_args()


def main():
    args = parse_args()
    questions = args.question or [DEFAULT_PROMPT]

    # MU-LLaMA's custom code lives in the repo's `llama` package.
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    model_path = args.model_path or os.path.join(repo, "mu_ckpts", "checkpoint.pth")
    llama_dir = args.llama_dir or os.path.join(repo, "mu_ckpts", "LLaMA")
    import llama  # noqa: E402

    print(f"Loading MU-LLaMA adapter={model_path} ...", flush=True)
    model = llama.load(
        model_path, llama_dir, mert_path=args.mert_path,
        knn=False, llama_type="7B", device=args.device,
    )
    model.eval()

    print(f"Loading audio: {args.audio}", flush=True)
    audio = load_audio(args.audio)
    inputs = {"Audio": [audio, 1]}

    for question in questions:
        prompts = [llama.format_prompt(question)]
        prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        with torch.cuda.amp.autocast():
            out = model.generate(
                inputs, prompts, max_gen_len=args.max_gen_len,
                temperature=0.6, top_p=0.8,
                cache_size=100, cache_t=20.0, cache_weight=0.0,
            )
        answer = out[0].strip()

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"A: {answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
