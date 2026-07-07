import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.WhisperTranscriptionService import (
    DiarizationSetupError,
    WhisperTranscriptionService,
    extract_transcript_speakers,
    match_speakers_to_profiles,
    merge_words_with_speakers,
    rename_transcript_speakers,
)


def make_service(**overrides):
    kwargs = {
        "model_size": "small",
        "device": "cpu",
        "compute_type": "auto",
    }
    kwargs.update(overrides)
    return WhisperTranscriptionService(**kwargs)


def fake_huggingface_hub(monkeypatch, cached_path):
    """Install a stub huggingface_hub whose cache lookup returns `cached_path`."""
    module = types.SimpleNamespace(try_to_load_from_cache=lambda *a, **k: cached_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


class TestMergeWordsWithSpeakers:
    def test_groups_consecutive_words_by_speaker(self):
        words = [(0.0, 0.5, "Hello"), (0.6, 1.0, "there"), (5.0, 5.5, "Hi")]
        turns = [(0.0, 2.0, "SPEAKER_00"), (4.0, 6.0, "SPEAKER_01")]
        lines, _ = merge_words_with_speakers(words, turns)
        assert lines == [(0.0, "Speaker 1", "Hello there"), (5.0, "Speaker 2", "Hi")]

    def test_speakers_numbered_by_first_appearance(self):
        # SPEAKER_01 talks first, so it becomes "Speaker 1".
        words = [(0.0, 0.5, "First"), (3.0, 3.5, "Second")]
        turns = [(0.0, 1.0, "SPEAKER_01"), (2.5, 4.0, "SPEAKER_00")]
        lines, label_map = merge_words_with_speakers(words, turns)
        assert [line[1] for line in lines] == ["Speaker 1", "Speaker 2"]
        assert label_map == {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}

    def test_splits_line_at_turn_boundary(self):
        words = [(0.0, 0.5, "Mine"), (1.0, 1.5, "yours"), (2.0, 2.5, "mine")]
        turns = [(0.0, 0.8, "A"), (0.9, 1.8, "B"), (1.9, 3.0, "A")]
        lines, _ = merge_words_with_speakers(words, turns)
        assert [(line[1], line[2]) for line in lines] == [
            ("Speaker 1", "Mine"),
            ("Speaker 2", "yours"),
            ("Speaker 1", "mine"),
        ]

    def test_word_with_no_overlap_gets_nearest_turn(self):
        # Word sits in a diarization gap; the closest turn edge is SPEAKER_00's.
        words = [(0.0, 0.5, "Hello"), (2.1, 2.3, "gap")]
        turns = [(0.0, 2.0, "SPEAKER_00"), (8.0, 9.0, "SPEAKER_01")]
        lines, label_map = merge_words_with_speakers(words, turns)
        assert lines == [(0.0, "Speaker 1", "Hello gap")]
        # SPEAKER_01 never got a line, so it must not appear in the label map.
        assert label_map == {"SPEAKER_00": "Speaker 1"}

    def test_empty_inputs(self):
        assert merge_words_with_speakers([], [(0.0, 1.0, "A")]) == ([], {})
        assert merge_words_with_speakers([(0.0, 1.0, "hi")], []) == ([], {})

    def test_skips_blank_and_untimed_words(self):
        words = [(0.0, 0.5, "  "), (None, 1.0, "lost"), (1.0, 1.5, "kept")]
        turns = [(0.0, 2.0, "A")]
        lines, _ = merge_words_with_speakers(words, turns)
        assert lines == [(1.0, "Speaker 1", "kept")]

    def test_identified_speakers_use_enrolled_names(self):
        words = [(0.0, 0.5, "Hello"), (5.0, 5.5, "Hi")]
        turns = [(0.0, 2.0, "SPEAKER_00"), (4.0, 6.0, "SPEAKER_01")]
        lines, label_map = merge_words_with_speakers(
            words, turns, speaker_names={"SPEAKER_01": "Andrew"}
        )
        # Unmatched speakers keep first-appearance numbering; matched ones are named.
        assert lines == [(0.0, "Speaker 1", "Hello"), (5.0, "Andrew", "Hi")]
        assert label_map == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Andrew"}


class TestTranscriptSpeakerHelpers:
    TRANSCRIPT = (
        "[00:00] Speaker 1: Hello there\n"
        "[00:05] Andrew: Hi\n"
        "no speaker on this line\n"
        "[00:12] Speaker 1: mentioning Speaker 2: inline stays untouched"
    )

    def test_extract_transcript_speakers(self):
        assert extract_transcript_speakers(self.TRANSCRIPT) == {"Speaker 1", "Andrew"}

    def test_extract_handles_empty(self):
        assert extract_transcript_speakers("") == set()
        assert extract_transcript_speakers(None) == set()

    def test_rename_preserves_timestamp_and_text(self):
        renamed = rename_transcript_speakers(self.TRANSCRIPT, {"Speaker 1": "Zoe"})
        assert renamed == (
            "[00:00] Zoe: Hello there\n"
            "[00:05] Andrew: Hi\n"
            "no speaker on this line\n"
            "[00:12] Zoe: mentioning Speaker 2: inline stays untouched"
        )

    def test_rename_unknown_label_is_noop(self):
        assert rename_transcript_speakers(self.TRANSCRIPT, {"Ghost": "Zoe"}) == self.TRANSCRIPT

    def test_rename_without_timestamps(self):
        assert rename_transcript_speakers("Speaker 1: hi", {"Speaker 1": "Zoe"}) == "Zoe: hi"


class TestMatchSpeakersToProfiles:
    def _profiles(self, *named_vecs):
        return [
            {"name": name, "embedding": np.asarray(vec, dtype=np.float32), "model_id": "m"}
            for name, vec in named_vecs
        ]

    def test_confident_match(self):
        emb = {"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)}
        profiles = self._profiles(("Andrew", [1.0, 0.0]), ("Zoe", [0.0, 1.0]))
        assert match_speakers_to_profiles(emb, profiles, 0.5, 0.05) == {"SPEAKER_00": "Andrew"}

    def test_below_threshold_stays_anonymous(self):
        emb = {"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)}
        profiles = self._profiles(("Zoe", [0.0, 1.0]))  # cosine 0
        assert match_speakers_to_profiles(emb, profiles, 0.5, 0.05) == {}

    def test_ambiguous_between_two_profiles_stays_anonymous(self):
        # Equidistant from both enrolled voices: must not pick either.
        emb = {"SPEAKER_00": np.array([0.7071, 0.7071], dtype=np.float32)}
        profiles = self._profiles(("Andrew", [1.0, 0.0]), ("Zoe", [0.0, 1.0]))
        assert match_speakers_to_profiles(emb, profiles, 0.5, 0.05) == {}

    def test_profile_claimed_once_by_most_confident_label(self):
        emb = {
            "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.array([0.9, 0.43589], dtype=np.float32),  # cosine 0.9 to Andrew
        }
        profiles = self._profiles(("Andrew", [1.0, 0.0]))
        assert match_speakers_to_profiles(emb, profiles, 0.5, 0.05) == {"SPEAKER_00": "Andrew"}

    def test_dimension_mismatch_skipped(self):
        emb = {"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)}
        profiles = self._profiles(("Old", [1.0, 0.0, 0.0]))
        assert match_speakers_to_profiles(emb, profiles, 0.5, 0.05) == {}

    def test_no_profiles(self):
        emb = {"SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32)}
        assert match_speakers_to_profiles(emb, [], 0.5, 0.05) == {}


class TestDiarizationPreflight:
    def test_enabled_without_token_or_cache_fails_before_whisper(self, tmp_path, monkeypatch):
        fake_huggingface_hub(monkeypatch, cached_path=None)
        audio = tmp_path / "rec.mp3"
        audio.write_bytes(b"\x00")
        service = make_service(diarization_enabled=True, hf_token=None)
        service._get_model = MagicMock()

        with pytest.raises(DiarizationSetupError, match="HF_TOKEN"):
            service.transcribe(str(audio))
        service._get_model.assert_not_called()

    def test_pipeline_access_failure_fails_before_whisper(self, tmp_path):
        # Token set but gated-model terms not accepted on HF: the cheap preflight
        # passes, so the eager pipeline load must fail before any Whisper work.
        audio = tmp_path / "rec.mp3"
        audio.write_bytes(b"\x00")
        service = make_service(diarization_enabled=True, hf_token="hf_x")
        service._get_model = MagicMock()
        service._get_diarization_pipeline = MagicMock(
            side_effect=DiarizationSetupError("Failed to load the diarization pipeline")
        )

        with pytest.raises(DiarizationSetupError, match="diarization pipeline"):
            service.transcribe(str(audio))
        service._get_model.assert_not_called()

    def test_enabled_without_token_but_cached_passes_preflight(self, monkeypatch):
        fake_huggingface_hub(monkeypatch, cached_path="/fake/hf_cache/config.yaml")
        service = make_service(diarization_enabled=True, hf_token=None)
        service._ensure_diarization_ready()  # must not raise

    def test_enabled_with_token_passes_preflight(self):
        service = make_service(diarization_enabled=True, hf_token="hf_x")
        service._ensure_diarization_ready()  # must not raise (no cache lookup needed)

    def test_disabled_never_checks_token(self, tmp_path):
        audio = tmp_path / "rec.mp3"
        audio.write_bytes(b"\x00")
        service = make_service(diarization_enabled=False, hf_token=None)

        segment = MagicMock(start=0.0, text=" Hello world ")
        info = MagicMock(language="en", language_probability=0.99)
        model = MagicMock()
        model.transcribe.return_value = (iter([segment]), info)
        service._get_model = MagicMock(return_value=model)

        assert service.transcribe(str(audio)) == "[00:00] Hello world"


class TestDiarizationPipelineLoad:
    def _install_fake_pyannote(self, monkeypatch, from_pretrained):
        torch_mod = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            device=lambda name: name,
            from_numpy=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        pyannote_pkg = types.ModuleType("pyannote")
        audio_mod = types.ModuleType("pyannote.audio")
        audio_mod.Pipeline = types.SimpleNamespace(from_pretrained=from_pretrained)
        pyannote_pkg.audio = audio_mod
        monkeypatch.setitem(sys.modules, "pyannote", pyannote_pkg)
        monkeypatch.setitem(sys.modules, "pyannote.audio", audio_mod)

    def test_load_failure_raises_with_setup_hint(self, monkeypatch):
        def failing_load(*a, **k):
            raise Exception("403 gated repo")

        self._install_fake_pyannote(monkeypatch, failing_load)
        service = make_service(diarization_enabled=True, hf_token="hf_x")
        with pytest.raises(DiarizationSetupError, match="HF_TOKEN"):
            service._get_diarization_pipeline()

    def test_load_success_moves_pipeline_to_device_and_caches(self, monkeypatch):
        pipeline = MagicMock()
        self._install_fake_pyannote(monkeypatch, lambda *a, **k: pipeline)
        service = make_service(diarization_enabled=True, hf_token="hf_x", diarization_device="cpu")

        assert service._get_diarization_pipeline() is pipeline
        pipeline.to.assert_called_once_with("cpu")
        assert service._get_diarization_pipeline() is pipeline  # cached, no reload


class TestDiarize:
    """_diarize must handle pyannote.audio 3.x ((Annotation, centroids) via the
    return_embeddings kwarg) and 4.x (DiarizeOutput wrapper, kwarg rejected)."""

    def _make_service_with_pipeline(self, monkeypatch, pipeline):
        torch_mod = types.SimpleNamespace(from_numpy=MagicMock(return_value=MagicMock()))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        service = make_service(diarization_enabled=True, hf_token="hf_x")
        service._diarization_pipeline = pipeline
        return service

    def _annotation(self):
        # Plain namespace, not MagicMock: a mock would auto-create
        # .speaker_diarization and defeat the 3.x/4.x detection.
        segment = types.SimpleNamespace(start=0.0, end=1.5)
        return types.SimpleNamespace(
            itertracks=lambda yield_label: [(segment, None, "SPEAKER_00")],
            labels=lambda: ["SPEAKER_00"],
        )

    def test_pyannote3_tuple_output_with_embeddings(self, monkeypatch):
        def pipeline(inputs, return_embeddings=False):
            assert return_embeddings is True
            return (self._annotation(), np.array([[3.0, 4.0]], dtype=np.float32))

        service = self._make_service_with_pipeline(monkeypatch, pipeline)
        turns, embeddings = service._diarize(MagicMock())
        assert turns == [(0.0, 1.5, "SPEAKER_00")]
        assert np.allclose(embeddings["SPEAKER_00"], [0.6, 0.8])  # L2-normalized

    def test_pyannote4_diarize_output_with_embeddings(self, monkeypatch):
        # A single-positional-arg lambda rejects the return_embeddings kwarg with
        # TypeError, exactly like a 4.x pipeline; centroids ride on the output object.
        wrapped = types.SimpleNamespace(
            speaker_diarization=self._annotation(),
            speaker_embeddings=np.array([[3.0, 4.0]], dtype=np.float32),
        )
        service = self._make_service_with_pipeline(monkeypatch, lambda inputs: wrapped)
        turns, embeddings = service._diarize(MagicMock())
        assert turns == [(0.0, 1.5, "SPEAKER_00")]
        assert np.allclose(embeddings["SPEAKER_00"], [0.6, 0.8])

    def test_plain_annotation_output_without_embeddings(self, monkeypatch):
        # 4.x legacy mode returns a bare Annotation: turns still work, no voiceprints.
        service = self._make_service_with_pipeline(
            monkeypatch, lambda inputs: self._annotation()
        )
        turns, embeddings = service._diarize(MagicMock())
        assert turns == [(0.0, 1.5, "SPEAKER_00")]
        assert embeddings == {}

    def test_zero_and_nan_centroids_skipped(self, monkeypatch):
        # 4.x zero-pads centroids for speakers the clustering couldn't embed and
        # may emit NaN rows; neither is a usable voiceprint.
        segment = types.SimpleNamespace(start=0.0, end=1.5)
        ann = types.SimpleNamespace(
            itertracks=lambda yield_label: [(segment, None, "SPEAKER_00")],
            labels=lambda: ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        )
        wrapped = types.SimpleNamespace(
            speaker_diarization=ann,
            speaker_embeddings=np.array(
                [[0.0, 0.0], [np.nan, 1.0], [1.0, 0.0]], dtype=np.float32
            ),
        )
        service = self._make_service_with_pipeline(monkeypatch, lambda inputs: wrapped)
        _, embeddings = service._diarize(MagicMock())
        assert list(embeddings) == ["SPEAKER_02"]

    def test_unsupported_output_raises_setup_error(self, monkeypatch):
        service = self._make_service_with_pipeline(monkeypatch, lambda inputs: object())
        with pytest.raises(DiarizationSetupError, match="unsupported output type"):
            service._diarize(MagicMock())


class TestTranscribeDetailedDiarized:
    def _make_diarized_service(self, tmp_path, monkeypatch, **service_overrides):
        """Service with stubbed Whisper + pyannote: two speakers, SPEAKER_00 with
        10s of speech (voiceprint [1, 0]), SPEAKER_01 with 1s (voiceprint [0, 1],
        below the persistence minimum)."""
        audio_file = tmp_path / "rec.mp3"
        audio_file.write_bytes(b"\x00")

        fw_audio = types.ModuleType("faster_whisper.audio")
        fw_audio.decode_audio = lambda p: np.zeros(16000, dtype=np.float32)
        fw_pkg = types.ModuleType("faster_whisper")
        fw_pkg.audio = fw_audio
        monkeypatch.setitem(sys.modules, "faster_whisper", fw_pkg)
        monkeypatch.setitem(sys.modules, "faster_whisper.audio", fw_audio)
        torch_mod = types.SimpleNamespace(from_numpy=MagicMock(return_value=MagicMock()))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        service = make_service(diarization_enabled=True, hf_token="hf_x", **service_overrides)

        w1 = types.SimpleNamespace(start=0.0, end=6.0, word=" Hello")
        w2 = types.SimpleNamespace(start=20.0, end=21.0, word=" Hi")
        segment = types.SimpleNamespace(words=[w1, w2])
        info = types.SimpleNamespace(language="en", language_probability=1.0)
        model = MagicMock()
        model.transcribe.return_value = (iter([segment]), info)
        service._get_model = MagicMock(return_value=model)

        seg_a = types.SimpleNamespace(start=0.0, end=10.0)
        seg_b = types.SimpleNamespace(start=20.0, end=21.0)
        ann = types.SimpleNamespace(
            itertracks=lambda yield_label: [
                (seg_a, None, "SPEAKER_00"),
                (seg_b, None, "SPEAKER_01"),
            ],
            labels=lambda: ["SPEAKER_00", "SPEAKER_01"],
        )
        out = types.SimpleNamespace(
            speaker_diarization=ann,
            speaker_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        service._diarization_pipeline = lambda inputs: out
        return service, audio_file

    def test_voiceprints_keyed_by_display_label_with_min_seconds_filter(
        self, tmp_path, monkeypatch
    ):
        service, audio_file = self._make_diarized_service(tmp_path, monkeypatch)

        result = service.transcribe_detailed(str(audio_file))
        assert result["transcript"] == "[00:00] Speaker 1: Hello\n[00:20] Speaker 2: Hi"
        assert [s["label"] for s in result["speakers"]] == ["Speaker 1"]
        speaker = result["speakers"][0]
        assert np.allclose(speaker["embedding"], [1.0, 0.0])
        assert speaker["speech_seconds"] == 10.0
        assert speaker["model_id"] == service._diarization_model

    def test_speaker_identification_names_matched_speakers(self, tmp_path, monkeypatch):
        service, audio_file = self._make_diarized_service(
            tmp_path, monkeypatch, speaker_id_enabled=True
        )
        profiles = [
            {
                "name": "Andrew",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "model_id": service._diarization_model,
            },
            {
                # Same voice under a stale embedding model: must be ignored.
                "name": "Ghost",
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "model_id": "some-old-model",
            },
        ]

        result = service.transcribe_detailed(str(audio_file), speaker_profiles=profiles)
        # SPEAKER_00 matches Andrew; SPEAKER_01 has no comparable profile and
        # keeps anonymous numbering (starting at 1 among unmatched speakers).
        assert result["transcript"] == "[00:00] Andrew: Hello\n[00:20] Speaker 1: Hi"
        # The persisted voiceprint is keyed the way the transcript displays it.
        assert [s["label"] for s in result["speakers"]] == ["Andrew"]

    def test_speaker_identification_off_by_default(self, tmp_path, monkeypatch):
        service, audio_file = self._make_diarized_service(tmp_path, monkeypatch)
        profiles = [
            {
                "name": "Andrew",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "model_id": service._diarization_model,
            }
        ]

        result = service.transcribe_detailed(str(audio_file), speaker_profiles=profiles)
        assert result["transcript"] == "[00:00] Speaker 1: Hello\n[00:20] Speaker 2: Hi"


class TestDeviceResolution:
    def test_explicit_device_wins(self):
        service = make_service(diarization_device="cpu")
        assert service._resolve_diarization_device() == "cpu"

    def test_auto_resolves_via_torch(self, monkeypatch):
        torch_mod = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        service = make_service(diarization_device="auto")
        assert service._resolve_diarization_device() == "cuda"
