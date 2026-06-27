#!/usr/bin/env python
"""
Ask questions about a music audio file with Gemma 4 (google/gemma-4-E4B-it).

Gemma 4 E2B/E4B are natively multimodal (text + image + audio + video) and
encoder-free. This script uses the audio path: pass an audio file + a question.

Usage:
    python gemma4_audio.py --audio samples/Grieg_30s.wav --device cuda:0 \
        -q "What instruments do you hear in this music?"

Notes:
    - Audio is limited to ~30 seconds per the model card; longer clips should be
      trimmed (see samples/Grieg_30s.wav).
    - Gemma recommends placing audio AFTER the text in the message content.
    - Requires accepting the Gemma license on HF + being logged in (huggingface-cli
      login). Needs `pillow` and `torchvision` (vision deps of the unified model).
"""
import argparse
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E4B-it"

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
    p = argparse.ArgumentParser(description="Gemma 4 music Q&A")
    p.add_argument("--audio", required=True, help="path to an audio file (<=30s)")
    p.add_argument(
        "-q", "--question", action="append", default=None,
        help="question about the audio (repeatable). Defaults to a detailed prompt.",
    )
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    return p.parse_args()


def main():
    args = parse_args()
    questions = args.question or [DEFAULT_PROMPT]

    print(f"Loading {MODEL_ID} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map=args.device,
    )
    model.eval()

    for question in questions:
        # Gemma recommends audio AFTER the text in the content list.
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "audio", "audio": args.audio},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        answer = processor.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )[0]

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"A: {answer.strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
