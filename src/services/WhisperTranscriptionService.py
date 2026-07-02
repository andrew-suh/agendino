import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DIARIZATION_SETUP_HINT = (
    "Set HF_TOKEN in .env (and accept the model terms on "
    "hf.co/pyannote/speaker-diarization-3.1 and hf.co/pyannote/segmentation-3.0), "
    "or set LOCAL_DIARIZATION_ENABLED=false."
)


class DiarizationSetupError(RuntimeError):
    """Diarization can't run due to configuration (missing token, gated-model terms
    not accepted, model not cached). Retrying won't help — Celery must not autoretry."""


def merge_words_with_speakers(words, turns):
    """Group timed words into speaker-labeled lines.

    Args:
        words: list of (start, end, text) from Whisper word timestamps
        turns: list of (start, end, raw_label) speaker turns from diarization

    Returns:
        list of (start_seconds, speaker_label, text) lines, where speaker labels
        are renumbered "Speaker 1", "Speaker 2", … in order of first appearance.
    """
    if not words or not turns:
        return []

    def speaker_for(w_start, w_end):
        best_label, best_overlap = None, 0.0
        for t_start, t_end, label in turns:
            overlap = min(w_end, t_end) - max(w_start, t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        if best_label is None:
            # No overlapping turn (e.g. a word in a diarization gap): nearest turn edge.
            mid = (w_start + w_end) / 2
            _, _, best_label = min(
                turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1]))
            )
        return best_label

    speaker_numbers: dict[str, int] = {}

    def display_name(raw_label):
        if raw_label not in speaker_numbers:
            speaker_numbers[raw_label] = len(speaker_numbers) + 1
        return f"Speaker {speaker_numbers[raw_label]}"

    lines = []
    current = None  # [start, raw_label, [texts]]
    for w_start, w_end, text in words:
        text = text.strip()
        if not text or w_start is None or w_end is None:
            continue
        raw_label = speaker_for(w_start, w_end)
        if current is not None and current[1] == raw_label:
            current[2].append(text)
        else:
            if current is not None:
                lines.append((current[0], display_name(current[1]), " ".join(current[2])))
            current = [w_start, raw_label, [text]]
    if current is not None:
        lines.append((current[0], display_name(current[1]), " ".join(current[2])))
    return lines


class WhisperTranscriptionService:
    """Transcribe audio locally using faster-whisper (CTranslate2 Whisper).

    Optionally labels speakers via a local pyannote diarization pipeline, producing
    the same "[MM:SS] Speaker N:" transcript format as Gemini transcription.
    """

    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        diarization_enabled: bool = False,
        hf_token: str | None = None,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        diarization_device: str = "auto",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._diarization_enabled = diarization_enabled
        self._hf_token = hf_token
        self._diarization_model = diarization_model
        self._diarization_device = diarization_device
        self._model = None  # lazy-loaded
        self._diarization_pipeline = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model '%s' (device=%s, compute_type=%s)…",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper model loaded.")
        return self._model

    def _ensure_diarization_ready(self) -> None:
        """Fail fast (before any transcription work) when diarization can't possibly run.

        HF_TOKEN is only needed for the first gated download — a token-less setup is
        fine as long as the pipeline weights are already in the local HF cache.
        """
        if self._diarization_pipeline is not None or self._hf_token:
            return

        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(self._diarization_model, "config.yaml")
        if isinstance(cached, str):
            return
        raise DiarizationSetupError(
            "Local diarization is enabled but HF_TOKEN is not set and the pyannote "
            f"models are not cached. {DIARIZATION_SETUP_HINT}"
        )

    def _resolve_diarization_device(self) -> str:
        if self._diarization_device != "auto":
            return self._diarization_device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_diarization_pipeline(self):
        if self._diarization_pipeline is None:
            try:
                import torch
                from pyannote.audio import Pipeline

                logger.info(
                    "Loading diarization pipeline '%s' (device=%s)…",
                    self._diarization_model,
                    self._diarization_device,
                )
                # No token kwarg: huggingface_hub reads the HF_TOKEN env var itself,
                # which keeps this compatible across pyannote token-kwarg renames.
                pipeline = Pipeline.from_pretrained(self._diarization_model)
                if pipeline is None:
                    raise RuntimeError(f"Pipeline.from_pretrained returned None for '{self._diarization_model}'")
                pipeline.to(torch.device(self._resolve_diarization_device()))
                self._diarization_pipeline = pipeline
                logger.info("Diarization pipeline loaded.")
            except Exception as e:
                raise DiarizationSetupError(
                    f"Failed to load the diarization pipeline '{self._diarization_model}': {e}. "
                    f"{DIARIZATION_SETUP_HINT}"
                ) from e
        return self._diarization_pipeline

    def _diarize(self, audio):
        """Run the diarization pipeline on a decoded waveform (float32 mono @16kHz)."""
        import torch

        pipeline = self._get_diarization_pipeline()
        waveform = torch.from_numpy(audio).unsqueeze(0)
        annotation = pipeline({"waveform": waveform, "sample_rate": 16000})
        return [
            (segment.start, segment.end, label)
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]

    def transcribe(self, audio_path: str, mime_type: str = "audio/mpeg") -> str:
        """Transcribe an audio file locally with Whisper and return formatted text."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._diarization_enabled:
            self._ensure_diarization_ready()
            # Load the pipeline before any Whisper work so access problems the cheap
            # preflight can't see (e.g. gated-model terms not accepted on HF) fail in
            # seconds instead of after a full transcription pass.
            self._get_diarization_pipeline()

        logger.info("Transcribing '%s' with local Whisper (%s)…", path.name, self._model_size)

        model = self._get_model()
        transcribe_kwargs = {"beam_size": 5, "vad_filter": True}
        # Distilled checkpoints (distil-*) are prone to repetition with the default
        # conditioning; faster-whisper recommends disabling it for them.
        if self._model_size.startswith("distil"):
            transcribe_kwargs["condition_on_previous_text"] = False

        if not self._diarization_enabled:
            segments, info = model.transcribe(str(path), **transcribe_kwargs)
            return self._format_plain(segments, info)

        # Diarized path: decode once (PyAV — no ffmpeg needed) and feed the same
        # waveform to both Whisper and pyannote; word timestamps drive attribution.
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(path))
        transcribe_kwargs["word_timestamps"] = True
        segments, info = model.transcribe(audio, **transcribe_kwargs)
        segments = list(segments)  # materialize the generator; reused on fallback

        logger.info(
            "Detected language: %s (probability %.2f)",
            info.language,
            info.language_probability,
        )
        logger.info("Diarizing '%s' with %s…", path.name, self._diarization_model)
        turns = self._diarize(audio)

        words = [
            (w.start, w.end, w.word)
            for segment in segments
            for w in (segment.words or [])
        ]
        speaker_lines = merge_words_with_speakers(words, turns)
        if not speaker_lines:
            # Legitimate data outcome (e.g. no speech turns found), not a setup error.
            logger.warning("Diarization found no speaker turns; emitting unlabeled transcript.")
            return self._format_segments(segments)

        transcript = "\n".join(
            f"[{self._format_timestamp(start)}] {speaker}: {text}"
            for start, speaker, text in speaker_lines
        )
        logger.info(
            "Whisper transcription complete (%d speaker lines, %d turns).",
            len(speaker_lines),
            len(turns),
        )
        return transcript

    def _format_plain(self, segments, info) -> str:
        logger.info(
            "Detected language: %s (probability %.2f)",
            info.language,
            info.language_probability,
        )
        transcript = self._format_segments(segments)
        logger.info("Whisper transcription complete (%d segments).", transcript.count("\n") + 1 if transcript else 0)
        return transcript

    def _format_segments(self, segments) -> str:
        lines = []
        for segment in segments:
            ts = self._format_timestamp(segment.start)
            text = segment.text.strip()
            if text:
                lines.append(f"[{ts}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"
