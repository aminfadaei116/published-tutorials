# Can a Language Model Hear? A Technical Tour of Audio & Music Understanding LLMs

*A hands-on comparison of five open audio-language models — Qwen2-Audio, Audio
Flamingo 3, Gemma 4, MOSS-Audio, and MU-LLaMA — on the exact same task: listen to
a piece of music and describe it. Every model below ships with runnable code in
[`code/`](code/) so you can reproduce the results yourself.*

---

## Why "audio LLMs" are interesting

Large language models reason over text. But a huge amount of human information —
speech, music, ambient sound — never gets written down. An **audio-language model
(audio LLM)** bolts an ear onto an LLM: it takes raw waveform *and* text as input
and produces text as output. Ask it "what instruments do you hear?" or "transcribe
this and summarize the mood," and it answers in natural language.

Music is the hardest stress test for these models. Speech recognition mostly needs
to map sound → words. Music understanding needs the model to reason about
*simultaneous* events — multiple instruments, harmony, rhythm, timbre, production —
and verbalize abstract perceptual qualities like "swung feel" or "melancholic but
gritty." It's where the differences between architectures show up most starkly.

This post walks through five open models that approach the problem in three
fundamentally different ways, gives you copy-pasteable code for each, and then puts
them head-to-head on the same audio clips.

---

## The three architectures

Almost every audio LLM is one of these shapes:

### 1. Encoder → adapter → frozen LLM
The dominant pattern. A pretrained **audio encoder** (often a Whisper or a
self-supervised music encoder like MERT) turns the waveform into a sequence of
embeddings. A small trainable **adapter / projection** maps those embeddings into
the LLM's token-embedding space, where they're spliced in alongside the text
tokens. The LLM itself is frozen or lightly fine-tuned. Cheap to train, modular,
but capped by the encoder's resolution.

*Examples here: Qwen2-Audio, Audio Flamingo 3, MU-LLaMA.*

### 2. Natively multimodal (encoder-free / unified)
No separate bolted-on encoder. Audio (and image, and video) are tokenized and fed
through the **same** transformer stack that handles text, trained jointly from
early on. One set of weights does everything. Tighter integration, but audio
quality is bounded by what a general-purpose model can spare for it.

*Example here: Gemma 4.*

### 3. Discrete audio tokens
The waveform is quantized into **discrete acoustic tokens** (think "words for
sound") by a neural audio codec, and the LLM treats them as just more vocabulary.
This makes the model symmetric: it can both *read* audio tokens (understanding) and
*write* them (generation/speech synthesis). Powerful and unifying, but the
tokenizer is now part of your fidelity budget.

*Example here: MOSS-Audio.*

Keep these three buckets in mind — they explain most of the behavioral differences
you'll see at the end.

---

## The contenders

| Model | Org | Params | Architecture | Audio encoder | Max audio | Env |
|---|---|---|---|---|---|---|
| **Qwen2-Audio** | Alibaba | 7B | Encoder + adapter | Whisper-large | ~30s/window | `persona` |
| **Audio Flamingo 3** | NVIDIA | ~8B | Encoder + adapter (cross-attn) | AF-Whisper | ~10 min (30s windows) | `persona` |
| **Gemma 4 E4B** | Google | ~4B eff. | Natively multimodal | — (encoder-free) | ~30s | `persona` |
| **MOSS-Audio** | OpenMOSS | 8B | Discrete audio tokens | neural codec | long | `moss-audio` |
| **MU-LLaMA** | Research | 7B | Encoder + adapter | MERT-v1-330M (music) | ~clip | `mu-llama` |

All five take a waveform plus a text question and return text. What differs is how
they hear, how fast they run, and how much they hallucinate. Let's run each one.

---

## 1. Qwen2-Audio — the reliable generalist

**What it is.** Alibaba's `Qwen2-Audio-7B-Instruct` pairs a Whisper-style audio
encoder with the Qwen2 7B LLM via an adapter. It was trained on a broad mix of
speech, sound, and music, and it's the most "just works" model of the group.

**The one gotcha.** You must decode the audio yourself with `librosa` at **16 kHz
mono** and hand the *samples* to the processor — the `audio_url` in the chat
template is only a placeholder. In `transformers` 5.x the keyword is `audio=` (it
used to be `audios=`).

```python
import librosa, torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

waveform, _ = librosa.load("song.wav", sr=16000, mono=True)   # decode yourself!
conversation = [{"role": "user", "content": [
    {"type": "audio", "audio_url": "song.wav"},               # placeholder only
    {"type": "text", "text": "What instruments do you hear?"},
]}]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, audio=[waveform], sampling_rate=16000,
                   return_tensors="pt", padding=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, inputs.input_ids.size(1):], skip_special_tokens=True)[0])
```

▶ Full script: [`code/qwen2_audio.py`](code/qwen2_audio.py) — run with
`python qwen2_audio.py --audio song.wav -q "your question"`.

---

## 2. Audio Flamingo 3 — the speed champion

**What it is.** NVIDIA's `audio-flamingo-3-hf` uses an AF-Whisper encoder feeding
the LLM through Flamingo-style cross-attention. It's built for long audio (up to
~10 minutes, internally chunked into 30s windows) and, as you'll see, it's
*dramatically* faster than everything else here.

**The one gotcha.** Load it in **`float32`**, not `bfloat16`. The audio tower keeps
some submodules (LayerNorm) in fp32, so a bf16 load throws dtype-mismatch errors.
The upside: the processor reads the file straight from the `path` field — no manual
`librosa` decode needed.

```python
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="cuda:0").eval()   # fp32!

conversation = [{"role": "user", "content": [
    {"type": "text", "text": "Describe the genre, tempo, and mood."},
    {"type": "audio", "path": "song.wav"},                              # processor loads it
]}]
inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

▶ Full script: [`code/audio_flamingo3.py`](code/audio_flamingo3.py).

---

## 3. Gemma 4 — the encoder-free native

**What it is.** Google's `gemma-4-E4B-it` is *natively* multimodal — text, image,
audio, and video all flow through one encoder-free transformer. There's no separate
audio tower; it's the same weights that do everything. Audio understanding is a
side effect of a general model, which makes it a fascinating point of comparison.

**The gotchas.** Two of them: (1) audio is capped at **~30 seconds** — trim longer
clips first; (2) Gemma wants the **text *before* the audio** in the content list
(the opposite of Qwen2-Audio). It's also gated: accept the license and
`huggingface-cli login`, and install `pillow` + `torchvision` (vision deps of the
unified model) even though you're only using audio.

```python
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="cuda:0").eval()

messages = [{"role": "user", "content": [
    {"type": "text", "text": "What instruments do you hear?"},   # text FIRST
    {"type": "audio", "audio": "song_30s.wav"},                  # <= 30 seconds
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt").to(model.device)
n = inputs["input_ids"].shape[-1]
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, n:], skip_special_tokens=True)[0])
```

▶ Full script: [`code/gemma4_audio.py`](code/gemma4_audio.py).

---

## 4. MOSS-Audio — the discrete-token unifier

**What it is.** OpenMOSS's `MOSS-Audio-8B-Instruct` represents audio as **discrete
tokens** from a neural codec, so the same model can both understand *and* generate
audio. It's not a plain `AutoModel`: it ships custom modeling code in its GitHub
repo and loads weights from a **local** directory with `trust_remote_code=True`.

**The gotchas.** It needs its own conda env (`transformers==4.57.1`,
`torch==2.9.1+cu128` — incompatible with the 5.x env the others use), the repo's
`src/` on `sys.path`, and a local weight download. We also swap the repo's
`torchaudio.load` for `soundfile`, because on torch 2.9 torchaudio routes through
torchcodec and needs FFmpeg shared libs that often aren't present.

```python
import sys, torch
sys.path.insert(0, "third_party/MOSS-Audio")          # provides the src/ package
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor

MODEL_PATH = "third_party/MOSS-Audio/weights/MOSS-Audio-8B-Instruct"   # local!
model = MossAudioModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, dtype="auto", device_map="cuda:0").eval()
processor = MossAudioProcessor.from_pretrained(
    MODEL_PATH, trust_remote_code=True, enable_time_marker=True)

# load mono audio at the processor's mel sample rate (see full script for soundfile loader)
inputs = processor(text="What instruments do you hear?",
                   audios=[raw_audio], return_tensors="pt").to(model.device)
inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
out = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                     temperature=1.0, top_p=1.0, top_k=50, use_cache=True)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

▶ Full script (incl. the `soundfile` loader): [`code/moss_audio.py`](code/moss_audio.py).

---

## 5. MU-LLaMA — the music specialist

**What it is.** The only model here built *specifically* for music. MU-LLaMA freezes
a **MERT-v1-330M** music encoder (a self-supervised model trained on music, running
at 24 kHz) and adapts its features into LLaMA-7B. In principle the most
"music-native" encoder of the bunch.

**The gotchas.** It's a research repo, not a Hub model: clone it, download LLaMA-7B
weights plus the MU-LLaMA adapter checkpoint, and add the repo's `llama` package to
`sys.path`. MERT uses `trust_remote_code` and writes module files, so you must point
`HF_MODULES_CACHE` at a writable directory first.

```python
import os, sys, torch, torchaudio
os.environ["HF_MODULES_CACHE"] = os.path.expanduser("~/.cache/hf_modules")
sys.path.insert(0, "third_party/MU-LLaMA/MU-LLaMA")        # provides `llama`
import llama

model = llama.load("mu_ckpts/checkpoint.pth", "mu_ckpts/LLaMA",
                   mert_path="m-a-p/MERT-v1-330M", knn=False,
                   llama_type="7B", device="cuda:0").eval()

wav, sr = torchaudio.load("song.wav")                      # MERT wants 24 kHz mono
wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 24000)
inputs = {"Audio": [torch.stack([wav.squeeze(0)], dim=0), 1]}
prompts = [model.tokenizer.encode(llama.format_prompt("What instruments do you hear?"),
                                  bos=True, eos=False)]
with torch.cuda.amp.autocast():
    out = model.generate(inputs, prompts, max_gen_len=512, temperature=0.6, top_p=0.8,
                         cache_size=100, cache_t=20.0, cache_weight=0.0)
print(out[0].strip())
```

▶ Full script: [`code/mu_llama.py`](code/mu_llama.py).

---

## Head-to-head: the benchmark

We ran all five models with the **same prompt** (a detailed 8-point "expert music
analyst" brief — see [`code/qwen2_audio.py`](code/qwen2_audio.py) or the
[full report](results/full_report.md)) over **10 clips spanning 10 GTZAN genres**
(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) on a
single GPU. The raw per-clip outputs are in [`results/`](results/).

### Speed & accuracy

| Model | Architecture | Avg sec / clip | Genre named correctly |
|---|---|---:|---:|
| **Audio Flamingo 3** | Encoder + adapter | **3.6** | **8 / 10** |
| **MU-LLaMA** | Encoder + adapter (MERT) | 1.6 | 2 / 10 |
| **MOSS-Audio** | Discrete tokens | 12.6 | 6 / 10 |
| **Qwen2-Audio** | Encoder + adapter | 15.4 | **8 / 10** |
| **Gemma 4** | Encoder-free native | 23.6 | 2 / 10 |

*(Genre accuracy = did the model's free-text description name the right genre or a
synonym. Latency includes generation of a full multi-paragraph analysis; it scales
with how much each model chooses to write.)*

Two results jump out:

- **Audio Flamingo 3 is the sweet spot.** Top-tier genre accuracy *and* ~4–7×
  faster than the other strong models. NVIDIA optimized it for exactly this.
- **The music-specialist encoder didn't win.** MU-LLaMA is blazing fast but its
  output is terse and often wrong on genre — a reminder that a great *encoder*
  (MERT) doesn't help if the *adapter + LLM* can't verbalize what it heard.

### Same clip, five models

Nothing illustrates the differences like one audio clip through all five. Here's a
**blues** track, prompt: *"give a detailed structured analysis."*

> **Qwen2-Audio** *(33s)* — "...features a prominent guitar solo with accompaniment
> from acoustic rhythm guitar and possibly harmonica... The genre is blues,
> specifically Chicago blues... tempo around 109 BPM... in the key of F minor...
> The production quality is low, indicating it may be an amateur or vintage
> recording." *(Thorough, hedged, mostly right — and note it commits to BPM/key.)*

> **Audio Flamingo 3** *(3s)* — "A blues track featuring a soulful male vocalist...
> guitar, piano, and drums... moderate 4/4... key of F minor... raw, authentic
> quality characteristic of traditional blues." *(Concise, correct, fast.)*

> **Gemma 4** *(36s)* — "...highly polished, modern-sounding track rooted in late
> 1990s/early 2000s R&B/Pop... programmed beats... synthesized bass..." *(Confident,
> detailed, fluent prose — and **wrong**: it heard R&B, not blues. The encoder-free
> model writes beautifully but mishears.)*

> **MOSS-Audio** *(11s)* — "A high-energy instrumental blues-rock piece... warm
> overdriven electric guitar riff... shuffle drum pattern... ~147 BPM in 4/4...
> major-key tonality." *(Vivid and structured — but invents a specific artist
> attribution, a classic discrete-token hallucination.)*

> **MU-LLaMA** *(2s)* — "The music is described as having a jazzy feel, with a
> strong emphasis on the trumpet and saxophone." *(One sentence, and wrong on both
> genre and instruments.)*

The pattern repeats on other clips. On a **jazz** sample, Gemma 4 confidently
analyzed a track it titled *"Neon Drive"* full of synthesizers and a drum machine,
while Qwen2-Audio and Audio Flamingo 3 correctly heard an acoustic jazz combo, and
MU-LLaMA labeled it "classical."

---

## What this tells us

1. **Architecture predicts personality.**
   - *Encoder + adapter* (Qwen2-Audio, AF3) models are the most reliable music
     listeners — a dedicated audio encoder is hard to beat for *perception*.
   - *Encoder-free native* (Gemma 4) writes the most fluent, well-structured prose
     but is the most likely to **confidently hallucinate** — its audio fidelity is
     bounded by a general-purpose model's spare capacity.
   - *Discrete-token* (MOSS-Audio) models are vivid and great for generation-style
     tasks, but quantization can invent plausible-sounding details (artist names,
     specific BPMs) that aren't there.

2. **A specialist encoder is necessary but not sufficient.** MU-LLaMA has the most
   music-specific front-end (MERT) yet the weakest descriptions — the adapter and
   LLM matter as much as the ears.

3. **Confidence ≠ correctness.** Every model states BPM, key, and era with total
   assurance. Some are right, some are fabrications. For any real application,
   **verify the structured claims** (BPM, key) with a signal-processing tool;
   trust the models for *vibe*, instrumentation, and genre gist.

4. **Pick for your constraint.** Need throughput and accuracy? Audio Flamingo 3.
   Want a Swiss-army multimodal model and can tolerate misses? Gemma 4. Building
   something that also *generates* audio? MOSS-Audio's token approach is the right
   shape.

---

## Reproduce it yourself

Everything is in this folder:

- [`code/`](code/) — a standalone, runnable script per model, plus
  [`code/SETUP.md`](code/SETUP.md) with the exact conda environments.
- [`results/`](results/) — the raw per-clip JSON for all five models and the
  [full report](results/full_report.md) with every description.

```bash
# example: one model, one clip, your own question
cd code
conda activate persona
python audio_flamingo3.py --audio /path/to/your/song.wav \
    -q "What instruments do you hear?" \
    -q "What is the tempo and mood?"
```

Each script loads the model once and answers every `-q` in sequence, so stacking
questions is cheap. Drop in your own audio and listen to what the machines hear.
