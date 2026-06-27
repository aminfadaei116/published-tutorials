#!/usr/bin/env python
"""
Ask questions about a music audio file with Qwen2-Audio-7B-Instruct.

Usage:
    python qwen2_audio.py --audio samples/Grieg_Lyric_Pieces_Kobold.ogg \
        --question "Describe this music. What instruments and mood do you hear?"

    # multiple questions in one run (model loads once):
    python qwen2_audio.py --audio samples/Grieg_Lyric_Pieces_Kobold.ogg \
        -q "What instrument is this?" -q "What is the tempo and mood?"
"""
import argparse
import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
# Qwen2-Audio's audio encoder (Whisper) expects 16 kHz mono.
TARGET_SR = 16000

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
    p = argparse.ArgumentParser(description="Qwen2-Audio music Q&A")
    p.add_argument("--audio", required=True, help="path to an audio file")
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
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    print(f"Loading audio: {args.audio}", flush=True)
    waveform, _ = librosa.load(args.audio, sr=TARGET_SR, mono=True)

    for question in questions:
        # Chat template with an audio placeholder. Qwen2-Audio reads the audio
        # from the `audio_url`; we pass decoded samples to the processor below.
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": args.audio},
                {"type": "text", "text": question},
            ]},
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text,
            audio=[waveform],          # transformers 5.x uses `audio` (was `audios`)
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        # strip the prompt tokens, keep only the newly generated answer
        generated = generated[:, inputs.input_ids.size(1):]
        answer = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print(f"A: {answer.strip()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
