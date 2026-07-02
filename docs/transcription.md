# Transcription

Convert audio recordings to text using cloud-based (Gemini) or local (Whisper) speech-to-text engines.

![Transcription](screenshots/transcription.png)

---

## Overview

AgenDino offers two transcription engines. You can choose between them per recording depending on your needs.

## Engine Comparison

| Feature | Gemini (Cloud) | Whisper (Local) |
|---------|---------------|-----------------|
| **Runs on** | Google Cloud | Your machine |
| **Speaker diarization** | ✅ Automatic | ✅ Optional (local pyannote) |
| **Speaker labels** | ✅ Yes | ✅ With `LOCAL_DIARIZATION_ENABLED` |
| **Timestamps** | ✅ Yes | ✅ Yes |
| **Long recordings** | ⚠️ May truncate | ✅ Full transcription |
| **Privacy** | Audio sent to Google | Fully offline |
| **First-use setup** | None | Model download (~500 MB for `small`) |
| **Speed** | Fast (cloud) | Depends on hardware |

## Using Gemini Transcription

1. Select a synced or uploaded recording.
2. Click the **Transcribe** button (microphone icon).
3. Gemini processes the audio and returns a transcript with speaker diarization, labels, and timestamps.
4. The transcript is saved to the database.

> **Model:** Gemini transcription uses `GEMINI_MODEL` by default. To run transcription on a
> different model than the rest of the app, set `GEMINI_TRANSCRIPTION_MODEL` in `.env` — e.g. a
> cheaper/faster Flash model for this high-volume call while `GEMINI_MODEL` stays on a stronger
> model for summaries. It falls back to `GEMINI_MODEL` when unset.

## Using Whisper Transcription

1. Select a recording.
2. Click the **dropdown arrow** next to the Transcribe button and choose **Whisper (local)**.
3. On first use, the Whisper model is downloaded automatically.
4. Transcription runs entirely on your machine - no audio is uploaded.

### Whisper Configuration

Configure Whisper via environment variables in `.env`:

| Variable | Default | Options |
|----------|---------|---------|
| `WHISPER_MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | `auto` (GPU if available, else CPU), `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `auto`, `int8`, `int8_float16`, `float16`, `float32` |

Larger models produce better accuracy but require more RAM and processing time. The `small` model is a good balance for most use cases.

### Local speaker diarization (optional)

By default Whisper transcripts have timestamps but no speaker labels. Setting
`LOCAL_DIARIZATION_ENABLED=true` runs a local [pyannote](https://github.com/pyannote/pyannote-audio)
pipeline alongside Whisper and produces the same `[MM:SS] Speaker N:` format as Gemini
transcription — the speaker-rename UI works on the result. Everything runs offline in the
celery worker; no audio leaves your machine.

**One-time setup** (the pyannote models are free but license-gated on Hugging Face):

1. Create a free token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope).
2. While logged in, accept the terms on
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
3. Set `HF_TOKEN=hf_...` and `LOCAL_DIARIZATION_ENABLED=true` in `.env`.

The models download on first use (cached in `settings/hf_cache`; the token isn't needed after
that). Setup problems — no token and no cached models, or a token whose account hasn't accepted
the model terms — make transcription **fail immediately with a setup message** (the diarization
pipeline is loaded before any Whisper work, and setup errors skip the usual task retries)
rather than silently producing an unlabeled transcript.

| Variable | Default | Notes |
|----------|---------|-------|
| `LOCAL_DIARIZATION_ENABLED` | `false` | Speaker labels for local Whisper |
| `HF_TOKEN` | — | Only needed until the models are cached |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | |
| `DIARIZATION_DEVICE` | follows `WHISPER_DEVICE` | `auto`, `cpu`, `cuda` — set `cpu` to keep VRAM flat while Whisper uses the GPU (in Docker this is the only split knob, since compose hardcodes `WHISPER_DEVICE=auto` on celery) |

**Cost:** ~2–3 GB extra RAM (or VRAM) while loaded, and extra processing time — a small
fraction of the audio duration on GPU, roughly 0.5–1.5× the audio duration on CPU.

### GPU acceleration

With `WHISPER_DEVICE=auto`, Whisper runs on an NVIDIA GPU when one is made available to the
container and falls back to CPU otherwise. In Docker, enabling the GPU is a one-file toggle —
see [Docker → GPU acceleration](docker.md). The GPU runs on the **celery** worker, where
transcription executes.

#### Choosing settings for your card

The key constraint is the GPU's **compute capability (CC)**: `float16` requires **CC ≥ 7.0**.
Older cards must use `int8` or `float32`. VRAM caps the model size. These are set in
`compose.gpu.yaml` via `WHISPER_COMPUTE_TYPE` and `WHISPER_MODEL_SIZE`.

| Card class | Example | CC | `WHISPER_COMPUTE_TYPE` | `WHISPER_MODEL_SIZE` | Notes |
|---|---|---|---|---|---|
| Modern (Turing/Ampere/Ada+) | RTX 3080, 10 GB | 8.6 | `float16` | `large-v3` (or `medium`) | Best speed; tensor-core FP16 |
| Mid VRAM, modern | 6–8 GB, CC ≥ 7.0 | ≥ 7.0 | `float16` or `int8_float16` | `medium` / `small` | Drop model size if VRAM-limited |
| Older (Maxwell/Pascal) | GTX 750 Ti, 2 GB | 5.0 | `int8` (or `float32`) | `small` / `base` | **No float16** (CC < 7.0); tiny VRAM; Maxwell support is dropped in CUDA 13 |

Check your card's compute capability on [NVIDIA's CUDA GPUs list](https://developer.nvidia.com/cuda/gpus).
If unsure, leave `WHISPER_COMPUTE_TYPE=auto` — CTranslate2 picks a type the detected device supports.

## Editing Transcripts

After transcription, you can edit the transcript text directly from the dashboard. Changes are saved to the database.

---

**Related:** [Summarization](summarization.md) · [Recording Management](recording-management.md)
