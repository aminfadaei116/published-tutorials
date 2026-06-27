#!/usr/bin/env python
"""
Ask questions about a music audio file with MOSS-Audio (OpenMOSS).

MOSS-Audio is NOT a plain HuggingFace AutoModel: it ships custom modeling code
in the OpenMOSS/MOSS-Audio GitHub repo (src/) and loads weights from a LOCAL
directory with trust_remote_code=True. This script adapts the repo's infer.py
into the same CLI as the other models here.

IMPORTANT — separate environment:
    MOSS-Audio pins transformers==4.57.1 / torch==2.9.1+cu128, which conflicts
    with the `persona` env (transformers 5.x). Run it in its own env:

        conda create -n moss-audio python=3.12 -y
        conda activate moss-audio
        conda install -c conda-forge "ffmpeg=7" -y
        cd third_party/MOSS-Audio
        pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
        hf download OpenMOSS-Team/MOSS-Audio-8B-Instruct \
            --local-dir ./weights/MOSS-Audio-8B-Instruct

Usage (from the repo root or anywhere, pointing --repo at the clone):
    python moss_audio.py \
        --model-path third_party/MOSS-Audio/weights/MOSS-Audio-8B-Instruct \
        --repo third_party/MOSS-Audio \
        --audio samples/Grieg_Lyric_Pieces_Kobold.wav \
        -q "What instruments do you hear in this music?"
"""
import argparse
import os
import sys

import torch

# Rich default prompt: ask for a thorough, structured musical breakdown.
DEFAULT_PROMPT = (
    "You are an expert music analyst. Listen to this music very carefully and "
    "give a detailed, structured analysis. Cover each of the following:\n"
    "1. Instrumentation: every instrument you can identify, and the role each plays "
    "(lead, accompaniment, bass, percussion, etc.).\n"
    "2. Genre and subgenre, plus any stylistic influences and the likely era or period.\n"
    "3. Tempo in BPM, the time signature, and the overall rhythmic feel (e.g. swung, "
    "driving, rubato, syncopated).\n"
    "4. Key and tonality (major/minor), and any notable harmonic progressions, "
    "modulations, or melodic motifs.\n"
    "5. Dynamics, articulation, and performance techniques (e.g. legato, staccato, "
    "crescendos, vibrato, pedal use).\n"
    "6. Mood, emotion, and tone — describe the feeling in detail and how it evolves "
    "over the course of the excerpt.\n"
    "7. Structure and form: identify sections and how the piece develops.\n"
    "8. Production and recording qualities: acoustic vs. electronic, live vs. studio, "
    "and the overall audio character.\n"
    "If there are vocals, also describe the voice type, delivery, language, and lyrics. "
    "Be specific, descriptive, and confident in your observations."
)


def load_audio(path, sample_rate):
    """Mono float32 numpy at `sample_rate`.

    Replaces the repo's torchaudio.load loader, which on torch 2.9 routes through
    torchcodec and needs FFmpeg shared libs that aren't present here. soundfile
    (libsndfile) reads wav/flac/ogg directly; resample with torch if needed.
    """
    import numpy as np
    import soundfile as sf
    import torchaudio

    waveform, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
    waveform = waveform.mean(axis=1)  # mono
    if sr != sample_rate:
        t = torch.from_numpy(waveform).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=sample_rate)
        waveform = t.squeeze(0).numpy()
    return np.ascontiguousarray(waveform, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="MOSS-Audio music Q&A")
    p.add_argument("--audio", required=True, help="path to an audio file")
    p.add_argument(
        "--model-path", required=True,
        help="local dir with downloaded MOSS-Audio weights (e.g. .../MOSS-Audio-8B-Instruct)",
    )
    p.add_argument(
        "--repo", default="third_party/MOSS-Audio",
        help="path to the cloned OpenMOSS/MOSS-Audio repo (provides the src/ package)",
    )
    p.add_argument(
        "-q", "--question", action="append", default=None,
        help="question about the audio (repeatable). Defaults to a detailed prompt.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    questions = args.question or [DEFAULT_PROMPT]

    # MOSS-Audio's custom code lives in the repo's src/ package.
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    print(f"Loading MOSS-Audio from {args.model_path} ...", flush=True)
    model = MossAudioModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map=args.device,
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    print(f"Loading audio: {args.audio}", flush=True)
    raw_audio = load_audio(args.audio, sample_rate=processor.config.mel_sr)

    for question in questions:
        inputs = processor(text=question, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(model.device)
        # The mel features must match the model dtype (mirrors the repo's infer.py).
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        answer = processor.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        )

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"A: {answer.strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
