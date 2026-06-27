#!/usr/bin/env python
"""
Ask questions about a music audio file with Audio Flamingo 3 (NVIDIA).

Uses the HF-native build `nvidia/audio-flamingo-3-hf` and the
`AudioFlamingo3ForConditionalGeneration` class (transformers >= 5.x).

Usage:
    python audio_flamingo3.py --audio samples/Grieg_Lyric_Pieces_Kobold.wav \
        -q "What instruments do you hear in this music?" \
        -q "Describe the genre, tempo, and mood."

Notes:
    - Supports WAV / MP3 / FLAC, up to ~10 min (processed in 30s windows).
    - The processor loads the audio file itself from the `path` field, so no
      manual librosa decode is needed.
"""
import argparse
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"

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


def parse_args():
    p = argparse.ArgumentParser(description="Audio Flamingo 3 music Q&A")
    p.add_argument("--audio", required=True, help="path to an audio file (wav/mp3/flac)")
    p.add_argument(
        "-q", "--question", action="append", default=None,
        help="question about the audio (repeatable). Defaults to a description prompt.",
    )
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    return p.parse_args()


def main():
    args = parse_args()
    questions = args.question or [DEFAULT_PROMPT]

    print(f"Loading {MODEL_ID} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Load in float32: the AF3 audio tower keeps some submodules (LayerNorm) in
    # fp32, so a bf16 load yields mixed-precision dtype mismatches. fp32 fits
    # easily on an H200.
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=args.device,
    )
    model.eval()

    for question in questions:
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "audio", "path": args.audio},
            ]},
        ]
        # AF3's processor loads the audio from `path` and tokenizes in one step.
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        answer = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"A: {answer.strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
