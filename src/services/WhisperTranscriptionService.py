import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# A speaker needs enough speech for a usable voiceprint; centroids built from less
# than this are too noisy to enroll or match against, so they aren't persisted.
MIN_SPEAKER_EMBEDDING_SECONDS = 5.0

# A transcript speaker line: optional indent, optional [MM:SS] timestamp, then
# "Name:" — mirrors the frontend's rename parser in static/dashboard.js.
_SPEAKER_LINE_RE = re.compile(r"^(\s*)(\[\d{1,2}:\d{2}\]\s*)?([A-Za-z0-9\s\(\)]+?):")

# Anonymous diarization display labels ("Speaker 1", "Speaker 2", …).
ANONYMOUS_SPEAKER_RE = re.compile(r"^Speaker \d+$")


def extract_transcript_speakers(transcript: str) -> set[str]:
    """Speaker labels currently appearing in a transcript's speaker lines."""
    speakers = set()
    for line in (transcript or "").split("\n"):
        match = _SPEAKER_LINE_RE.match(line)
        if match:
            speakers.add(match.group(3).strip())
    return speakers


def rename_transcript_speakers(transcript: str, renames: dict[str, str]) -> str:
    """Apply {old_label: new_label} to a transcript's speaker lines, preserving
    indentation, timestamps, and the spoken text (server-side counterpart of the
    frontend rename editor)."""
    def rename_line(line):
        match = _SPEAKER_LINE_RE.match(line)
        if not match:
            return line
        new_label = renames.get(match.group(3).strip())
        if not new_label:
            return line
        return f"{match.group(1)}{match.group(2) or ''}{new_label}:{line[match.end():]}"

    return "\n".join(rename_line(line) for line in (transcript or "").split("\n"))


DIARIZATION_SETUP_HINT = (
    "Set HF_TOKEN in .env and accept the model terms on "
    "hf.co/pyannote/speaker-diarization-3.1, hf.co/pyannote/segmentation-3.0, and "
    "hf.co/pyannote/speaker-diarization-community-1 (pyannote.audio 4.x silently loads "
    "community-1 in place of the deprecated 3.1 pipeline), or set "
    "LOCAL_DIARIZATION_ENABLED=false."
)


class DiarizationSetupError(RuntimeError):
    """Diarization can't run due to configuration or environment (missing token,
    gated-model terms not accepted, model not cached, incompatible pyannote.audio
    version). Retrying won't help — Celery must not autoretry."""


def match_speakers_to_profiles(embeddings_by_label, profiles, threshold, margin):
    """Greedy cosine matching of diarized voiceprints against enrolled profiles.

    Both sides must be L2-normalized (dot product == cosine similarity). A raw
    label is matched to its single best profile only when the similarity clears
    `threshold` AND beats the second-best profile by at least `margin` (ambiguity
    between two enrolled people must never be resolved by a coin flip). Matches
    are assigned in descending-similarity order and each profile can claim only
    one label; everything else stays anonymous — a wrong name in a transcript is
    worse than no name.

    Returns {raw_label: profile_name} for the confident matches.
    """
    candidates = []
    for label, vec in embeddings_by_label.items():
        sims = []
        for idx, profile in enumerate(profiles):
            pvec = np.asarray(profile["embedding"], dtype=np.float32)
            if pvec.shape != vec.shape:
                continue
            sims.append((float(np.dot(vec, pvec)), idx))
        if not sims:
            continue
        sims.sort(reverse=True)
        best_sim, best_idx = sims[0]
        if best_sim < threshold:
            continue
        if len(sims) > 1 and best_sim - sims[1][0] < margin:
            continue
        candidates.append((best_sim, label, best_idx))

    matches = {}
    used_profiles = set()
    for sim, label, idx in sorted(candidates, reverse=True):
        if idx in used_profiles:
            # Best profile already claimed by a more confident label; leaving this
            # label anonymous beats falling back to a weaker candidate.
            continue
        matches[label] = profiles[idx]["name"]
        used_profiles.add(idx)
    return matches


def merge_words_with_speakers(words, turns, speaker_names=None):
    """Group timed words into speaker-labeled lines.

    Args:
        words: list of (start, end, text) from Whisper word timestamps
        turns: list of (start, end, raw_label) speaker turns from diarization
        speaker_names: optional {raw_label: person_name} from speaker
            identification; matched labels display the person's name

    Returns:
        (lines, label_map) where lines is a list of (start_seconds, speaker_label,
        text) with unmatched speaker labels numbered "Speaker 1", "Speaker 2", …
        in order of first appearance, and label_map maps each raw diarization
        label that appeared to its display label (used to key stored voiceprints
        the same way the transcript names speakers).
    """
    if not words or not turns:
        return [], {}

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
    resolved: dict[str, str] = {}  # raw label -> display label (name or "Speaker N")

    def display_name(raw_label):
        if raw_label not in resolved:
            if speaker_names and raw_label in speaker_names:
                resolved[raw_label] = speaker_names[raw_label]
            else:
                speaker_numbers[raw_label] = len(speaker_numbers) + 1
                resolved[raw_label] = f"Speaker {speaker_numbers[raw_label]}"
        return resolved[raw_label]

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
    return lines, dict(resolved)


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
        speaker_id_enabled: bool = False,
        speaker_id_threshold: float = 0.5,
        speaker_id_margin: float = 0.05,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._diarization_enabled = diarization_enabled
        self._hf_token = hf_token
        self._diarization_model = diarization_model
        self._diarization_device = diarization_device
        self._speaker_id_enabled = speaker_id_enabled
        self._speaker_id_threshold = speaker_id_threshold
        self._speaker_id_margin = speaker_id_margin
        self._model = None  # lazy-loaded
        self._diarization_pipeline = None  # lazy-loaded

    @property
    def speaker_id_enabled(self) -> bool:
        """True when enrolled voice profiles should be loaded and matched."""
        return self._speaker_id_enabled and self._diarization_enabled

    @property
    def speaker_id_threshold(self) -> float:
        return self._speaker_id_threshold

    @property
    def speaker_id_margin(self) -> float:
        return self._speaker_id_margin

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
        """Run the diarization pipeline on a decoded waveform (float32 mono @16kHz).

        Returns (turns, embeddings_by_label): turns is a list of (start, end,
        raw_label); embeddings_by_label maps raw labels to L2-normalized float32
        centroid voiceprints, empty when the installed pyannote can't provide them.
        """
        import torch

        pipeline = self._get_diarization_pipeline()
        waveform = torch.from_numpy(audio).unsqueeze(0)
        inputs = {"waveform": waveform, "sample_rate": 16000}
        embeddings = None
        try:
            # pyannote.audio 3.x: returns (Annotation, centroids) with this kwarg.
            # 4.x rejects it at signature binding — before any heavy compute — so
            # the fallback below does not run diarization twice.
            result = pipeline(inputs, return_embeddings=True)
        except TypeError:
            result = pipeline(inputs)

        if isinstance(result, tuple) and len(result) == 2:
            annotation, embeddings = result
        else:
            # pyannote.audio 4.x returns a DiarizeOutput (legacy mode: a bare
            # Annotation, without embeddings).
            annotation = getattr(result, "speaker_diarization", result)
            embeddings = getattr(result, "speaker_embeddings", None)

        if not hasattr(annotation, "itertracks"):
            raise DiarizationSetupError(
                f"Diarization returned an unsupported output type "
                f"'{type(result).__name__}' — the installed pyannote.audio version is "
                "incompatible with this code. Pin pyannote.audio to a supported "
                "version (3.x or 4.x) and rebuild."
            )
        turns = [
            (segment.start, segment.end, label)
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]

        # Both 3.x and 4.x order centroid rows by annotation.labels(). Rows may be
        # zero-padded (4.x, speakers the clustering couldn't embed) or NaN — skip those.
        embeddings_by_label = {}
        if embeddings is not None:
            for i, label in enumerate(annotation.labels()):
                if i >= len(embeddings):
                    break
                vec = np.asarray(embeddings[i], dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vec))
                if not np.isfinite(vec).all() or norm == 0.0:
                    continue
                embeddings_by_label[str(label)] = vec / norm
        return turns, embeddings_by_label

    def _embedding_model_id(self) -> str:
        """Identifier stamped on stored voiceprints; embeddings from different
        models aren't comparable, so matching must be scoped to this id."""
        emb = getattr(self._diarization_pipeline, "embedding", None)
        return emb if isinstance(emb, str) else self._diarization_model

    def _speaker_profiles(self, turns, embeddings_by_label, label_map) -> list[dict]:
        """Voiceprints keyed by display label ("Speaker N"), ready to persist."""
        speech_seconds: dict[str, float] = {}
        for start, end, raw_label in turns:
            speech_seconds[raw_label] = speech_seconds.get(raw_label, 0.0) + max(0.0, end - start)

        model_id = self._embedding_model_id()
        profiles = []
        for raw_label, display in label_map.items():
            vec = embeddings_by_label.get(raw_label)
            seconds = speech_seconds.get(raw_label, 0.0)
            if vec is None or seconds < MIN_SPEAKER_EMBEDDING_SECONDS:
                continue
            profiles.append({
                "label": display,
                "embedding": vec,
                "model_id": model_id,
                "speech_seconds": round(seconds, 2),
            })
        return profiles

    def transcribe(self, audio_path: str, mime_type: str = "audio/mpeg") -> str:
        """Transcribe an audio file locally with Whisper and return formatted text."""
        return self.transcribe_detailed(audio_path, mime_type=mime_type)["transcript"]

    def transcribe_detailed(
        self, audio_path: str, mime_type: str = "audio/mpeg", speaker_profiles: list[dict] | None = None
    ) -> dict:
        """Transcribe and also return per-speaker voiceprints for speaker enrollment.

        speaker_profiles: enrolled voice profiles [{"name", "embedding", "model_id"}];
        when speaker identification is enabled, confidently matched speakers appear
        in the transcript under their enrolled names instead of "Speaker N".

        Returns {"transcript": str, "speakers": [{"label", "embedding", "model_id",
        "speech_seconds"}]} — "speakers" is keyed by the transcript's display labels
        and is empty when diarization is disabled, found no speakers, or the
        installed pyannote can't provide centroid embeddings.
        """
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
            return {"transcript": self._format_plain(segments, info), "speakers": []}

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
        turns, embeddings_by_label = self._diarize(audio)

        speaker_names = {}
        if self.speaker_id_enabled and speaker_profiles and embeddings_by_label:
            model_id = self._embedding_model_id()
            usable = [p for p in speaker_profiles if p.get("model_id") == model_id]
            speaker_names = match_speakers_to_profiles(
                embeddings_by_label, usable, self._speaker_id_threshold, self._speaker_id_margin
            )
            skipped = len(speaker_profiles) - len(usable)
            if skipped:
                logger.warning(
                    "Ignored %d voice profile(s) from a different embedding model — "
                    "re-enroll them to keep using speaker identification.",
                    skipped,
                )
            if speaker_names:
                logger.info(
                    "Identified speakers: %s",
                    ", ".join(sorted(speaker_names.values())),
                )

        words = [
            (w.start, w.end, w.word)
            for segment in segments
            for w in (segment.words or [])
        ]
        speaker_lines, label_map = merge_words_with_speakers(words, turns, speaker_names=speaker_names)
        if not speaker_lines:
            # Legitimate data outcome (e.g. no speech turns found), not a setup error.
            logger.warning("Diarization found no speaker turns; emitting unlabeled transcript.")
            return {"transcript": self._format_segments(segments), "speakers": []}

        transcript = "\n".join(
            f"[{self._format_timestamp(start)}] {speaker}: {text}"
            for start, speaker, text in speaker_lines
        )
        speakers = self._speaker_profiles(turns, embeddings_by_label, label_map)
        logger.info(
            "Whisper transcription complete (%d speaker lines, %d turns, %d voiceprints).",
            len(speaker_lines),
            len(turns),
            len(speakers),
        )
        return {"transcript": transcript, "speakers": speakers}

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
