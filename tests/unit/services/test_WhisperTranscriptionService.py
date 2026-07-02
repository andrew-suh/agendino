import sys
import types
from unittest.mock import MagicMock

import pytest

from services.WhisperTranscriptionService import (
    DiarizationSetupError,
    WhisperTranscriptionService,
    merge_words_with_speakers,
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
        lines = merge_words_with_speakers(words, turns)
        assert lines == [(0.0, "Speaker 1", "Hello there"), (5.0, "Speaker 2", "Hi")]

    def test_speakers_numbered_by_first_appearance(self):
        # SPEAKER_01 talks first, so it becomes "Speaker 1".
        words = [(0.0, 0.5, "First"), (3.0, 3.5, "Second")]
        turns = [(0.0, 1.0, "SPEAKER_01"), (2.5, 4.0, "SPEAKER_00")]
        lines = merge_words_with_speakers(words, turns)
        assert [line[1] for line in lines] == ["Speaker 1", "Speaker 2"]

    def test_splits_line_at_turn_boundary(self):
        words = [(0.0, 0.5, "Mine"), (1.0, 1.5, "yours"), (2.0, 2.5, "mine")]
        turns = [(0.0, 0.8, "A"), (0.9, 1.8, "B"), (1.9, 3.0, "A")]
        lines = merge_words_with_speakers(words, turns)
        assert [(line[1], line[2]) for line in lines] == [
            ("Speaker 1", "Mine"),
            ("Speaker 2", "yours"),
            ("Speaker 1", "mine"),
        ]

    def test_word_with_no_overlap_gets_nearest_turn(self):
        # Word sits in a diarization gap; the closest turn edge is SPEAKER_00's.
        words = [(0.0, 0.5, "Hello"), (2.1, 2.3, "gap")]
        turns = [(0.0, 2.0, "SPEAKER_00"), (8.0, 9.0, "SPEAKER_01")]
        lines = merge_words_with_speakers(words, turns)
        assert lines == [(0.0, "Speaker 1", "Hello gap")]

    def test_empty_inputs(self):
        assert merge_words_with_speakers([], [(0.0, 1.0, "A")]) == []
        assert merge_words_with_speakers([(0.0, 1.0, "hi")], []) == []

    def test_skips_blank_and_untimed_words(self):
        words = [(0.0, 0.5, "  "), (None, 1.0, "lost"), (1.0, 1.5, "kept")]
        turns = [(0.0, 2.0, "A")]
        assert merge_words_with_speakers(words, turns) == [(1.0, "Speaker 1", "kept")]


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


class TestDeviceResolution:
    def test_explicit_device_wins(self):
        service = make_service(diarization_device="cpu")
        assert service._resolve_diarization_device() == "cpu"

    def test_auto_resolves_via_torch(self, monkeypatch):
        torch_mod = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        service = make_service(diarization_device="auto")
        assert service._resolve_diarization_device() == "cuda"
