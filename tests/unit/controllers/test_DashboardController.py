import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from controllers.DashboardController import DashboardController
from models.DBRecording import DBRecording
from services.WhisperTranscriptionService import DiarizationSetupError


@pytest.fixture
def mock_services(tmp_path):
    """Create a DashboardController with all dependencies mocked."""
    # Create a minimal template directory so Jinja2Templates doesn't fail
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "home.html").write_text("<html></html>")

    sqlite_db = MagicMock()
    local_repo = MagicMock()
    system_prompts_repo = MagicMock()
    task_generation_service = MagicMock()
    transcription_service = MagicMock()
    summarization_service = MagicMock()
    whisper_service = MagicMock()

    controller = DashboardController(
        sqlite_db_repository=sqlite_db,
        local_recordings_repository=local_repo,
        transcription_service=transcription_service,
        system_prompts_repository=system_prompts_repo,
        template_path=str(template_dir),
        publish_services={},
        task_generation_service=task_generation_service,
        summarization_service=summarization_service,
        whisper_transcription_service=whisper_service,
    )

    return {
        "controller": controller,
        "sqlite_db": sqlite_db,
        "local_repo": local_repo,
        "transcription_service": transcription_service,
        "summarization_service": summarization_service,
        "system_prompts_repo": system_prompts_repo,
        "task_generation_service": task_generation_service,
        "whisper_service": whisper_service,
    }


class TestDashboardControllerBareName:
    def test_strips_single_hda(self):
        assert DashboardController._bare_name("2026Mar27-094938-Wip01.hda") == "2026Mar27-094938-Wip01"

    def test_strips_double_hda(self):
        # os.path.splitext only strips the last extension; this is expected
        assert DashboardController._bare_name("file.hda.hda") == "file.hda"

    def test_no_extension(self):
        assert DashboardController._bare_name("2026Mar27-094938-Wip01") == "2026Mar27-094938-Wip01"

    def test_empty_string(self):
        assert DashboardController._bare_name("") == ""

    def test_other_extension_untouched(self):
        assert DashboardController._bare_name("file.txt") == "file.txt"

    def test_strips_mp3(self):
        assert DashboardController._bare_name("recording.mp3") == "recording"

    def test_strips_wav(self):
        assert DashboardController._bare_name("my-meeting.wav") == "my-meeting"

    def test_strips_m4a(self):
        assert DashboardController._bare_name("audio.m4a") == "audio"

    def test_strips_flac(self):
        assert DashboardController._bare_name("lossless.flac") == "lossless"


class TestDashboardControllerParseRecordingDatetime:
    def test_valid_datetime(self):
        result = DashboardController._parse_recording_datetime("2026Mar27-094938-Wip01")
        assert result == "2026-03-27 09:49:38"

    def test_valid_april(self):
        result = DashboardController._parse_recording_datetime("2026Apr01-152300-Rec13")
        assert result == "2026-04-01 15:23:00"

    def test_invalid_name(self):
        result = DashboardController._parse_recording_datetime("not-a-valid-name")
        assert result is None

    def test_single_segment(self):
        result = DashboardController._parse_recording_datetime("noparts")
        assert result is None


class TestDashboardControllerTranscript:
    def test_get_transcript_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = "Hello transcript"

        result = ctrl.get_transcript("2026Mar27-094938-Wip01")
        assert result["ok"] is True
        assert result["transcript"] == "Hello transcript"

    def test_get_transcript_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = None

        result = ctrl.get_transcript("nonexistent")
        assert result["ok"] is False
        assert "No transcript" in result["error"]

    def test_get_transcript_strips_hda(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = "text"

        ctrl.get_transcript("2026Mar27-094938-Wip01.hda")
        mock_services["sqlite_db"].get_transcript.assert_called_with("2026Mar27-094938-Wip01")

    def test_update_transcript_success(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].update_transcript.return_value = True

        result = ctrl.update_transcript("2026Mar27-094938-Wip01.hda", "edited")
        assert result["ok"] is True
        assert result["transcript"] == "edited"
        mock_services["sqlite_db"].update_transcript.assert_called_with("2026Mar27-094938-Wip01", "edited")

    def test_update_transcript_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].update_transcript.return_value = False

        result = ctrl.update_transcript("ghost", "edited")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()


class TestDashboardControllerTranscribeRecording:
    def test_local_file_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = False

        result = ctrl.transcribe_recording("test")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_returns_cached_transcript(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1, name="test", label="Test", duration=10, created_at=datetime.now(), transcript="cached text"
        )

        result = ctrl.transcribe_recording("test")
        assert result["ok"] is True
        assert result["cached"] is True
        assert result["transcript"] == "cached text"

    def test_transcription_success(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["transcription_service"].transcribe.return_value = "new transcript"

        result = ctrl.transcribe_recording("test")
        assert result["ok"] is True
        assert result["cached"] is False
        assert result["transcript"] == "new transcript"
        mock_services["sqlite_db"].save_transcript.assert_called_once_with("test", "new transcript")

    def test_transcription_failure(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["transcription_service"].transcribe.side_effect = RuntimeError("API error")

        result = ctrl.transcribe_recording("test")
        assert result["ok"] is False
        assert "Transcription failed" in result["error"]

    def test_whisper_success_persists_voiceprints(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        speakers = [
            {"label": "Speaker 1", "embedding": [0.6, 0.8], "model_id": "m", "speech_seconds": 9.0}
        ]
        mock_services["whisper_service"].transcribe_detailed.return_value = {
            "transcript": "whisper text",
            "speakers": speakers,
        }

        result = ctrl.transcribe_recording("test", engine="whisper")
        assert result["ok"] is True
        assert result["transcript"] == "whisper text"
        mock_services["sqlite_db"].save_transcript.assert_called_once_with("test", "whisper text")
        mock_services["sqlite_db"].save_recording_speakers.assert_called_once_with("test", speakers)

    def test_whisper_voiceprint_save_failure_does_not_fail_transcription(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["whisper_service"].transcribe_detailed.return_value = {
            "transcript": "whisper text",
            "speakers": [{"label": "Speaker 1", "embedding": [1.0], "model_id": "m"}],
        }
        mock_services["sqlite_db"].save_recording_speakers.side_effect = RuntimeError("db locked")

        result = ctrl.transcribe_recording("test", engine="whisper")
        assert result["ok"] is True
        assert result["transcript"] == "whisper text"

    def test_whisper_without_speakers_skips_voiceprint_persistence(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["whisper_service"].transcribe_detailed.return_value = {
            "transcript": "plain text",
            "speakers": [],
        }

        result = ctrl.transcribe_recording("test", engine="whisper")
        assert result["ok"] is True
        mock_services["sqlite_db"].save_recording_speakers.assert_not_called()

    def _whisper_ready(self, mock_services):
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["whisper_service"].transcribe_detailed.return_value = {
            "transcript": "t",
            "speakers": [],
        }

    def test_whisper_passes_profiles_when_speaker_id_enabled(self, mock_services):
        ctrl = mock_services["controller"]
        self._whisper_ready(mock_services)
        mock_services["whisper_service"].speaker_id_enabled = True
        profiles = [{"name": "Andrew", "embedding": [1.0], "model_id": "m"}]
        mock_services["sqlite_db"].get_speaker_profiles.return_value = profiles

        assert ctrl.transcribe_recording("test", engine="whisper")["ok"] is True
        kwargs = mock_services["whisper_service"].transcribe_detailed.call_args.kwargs
        assert kwargs["speaker_profiles"] == profiles

    def test_whisper_skips_profile_load_when_speaker_id_disabled(self, mock_services):
        ctrl = mock_services["controller"]
        self._whisper_ready(mock_services)
        mock_services["whisper_service"].speaker_id_enabled = False

        assert ctrl.transcribe_recording("test", engine="whisper")["ok"] is True
        mock_services["sqlite_db"].get_speaker_profiles.assert_not_called()
        kwargs = mock_services["whisper_service"].transcribe_detailed.call_args.kwargs
        assert kwargs["speaker_profiles"] == []

    def test_whisper_profile_load_failure_does_not_fail_transcription(self, mock_services):
        ctrl = mock_services["controller"]
        self._whisper_ready(mock_services)
        mock_services["whisper_service"].speaker_id_enabled = True
        mock_services["sqlite_db"].get_speaker_profiles.side_effect = RuntimeError("db locked")

        assert ctrl.transcribe_recording("test", engine="whisper")["ok"] is True
        kwargs = mock_services["whisper_service"].transcribe_detailed.call_args.kwargs
        assert kwargs["speaker_profiles"] == []

    def test_diarization_setup_error_propagates(self, mock_services):
        # Not swallowed into the error dict: the Celery layer needs the exception
        # type to skip autoretry for non-recoverable setup problems.
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["sqlite_db"].get_recording_by_name.return_value = None
        mock_services["local_repo"].get_path.return_value = "/path/to/test.hda"
        mock_services["transcription_service"].transcribe.side_effect = DiarizationSetupError(
            "accept the model terms"
        )

        with pytest.raises(DiarizationSetupError, match="accept the model terms"):
            ctrl.transcribe_recording("test")


class TestDashboardControllerEnrollSpeaker:
    def _with_voiceprint(self, mock_services, embedding=(0.6, 0.8), model_id="m1"):
        mock_services["sqlite_db"].get_recording_speakers.return_value = [
            {
                "label": "Speaker 1",
                "embedding": np.array(embedding, dtype=np.float32),
                "model_id": model_id,
                "speech_seconds": 10.0,
            }
        ]

    def test_creates_new_profile_with_normalized_embedding(self, mock_services):
        ctrl = mock_services["controller"]
        self._with_voiceprint(mock_services, embedding=(3.0, 4.0))
        mock_services["sqlite_db"].get_speaker_profile_by_name.return_value = None

        result = ctrl.enroll_speaker("test.hda", "Speaker 1", "Andrew")
        assert result["ok"] is True
        assert result["enrollment_count"] == 1
        name, vec, model_id = mock_services["sqlite_db"].insert_speaker_profile.call_args[0]
        assert name == "Andrew"
        assert np.allclose(vec, [0.6, 0.8])
        assert model_id == "m1"
        # .hda suffix stripped before the voiceprint lookup
        mock_services["sqlite_db"].get_recording_speakers.assert_called_once_with("test")

    def test_repeat_enrollment_running_mean(self, mock_services):
        ctrl = mock_services["controller"]
        self._with_voiceprint(mock_services, embedding=(0.0, 1.0))
        mock_services["sqlite_db"].get_speaker_profile_by_name.return_value = {
            "id": 5,
            "name": "Andrew",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "model_id": "m1",
            "enrollment_count": 1,
        }

        result = ctrl.enroll_speaker("test", "Speaker 1", "Andrew")
        assert result["ok"] is True
        assert result["enrollment_count"] == 2
        profile_id, vec, model_id, count = mock_services["sqlite_db"].update_speaker_profile.call_args[0]
        assert profile_id == 5
        assert np.allclose(vec, [0.7071, 0.7071], atol=1e-3)  # mean of two units, renormalized
        assert count == 2

    def test_model_change_restarts_profile(self, mock_services):
        ctrl = mock_services["controller"]
        self._with_voiceprint(mock_services, embedding=(0.0, 1.0), model_id="new-model")
        mock_services["sqlite_db"].get_speaker_profile_by_name.return_value = {
            "id": 5,
            "name": "Andrew",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "model_id": "old-model",
            "enrollment_count": 4,
        }

        result = ctrl.enroll_speaker("test", "Speaker 1", "Andrew")
        assert result["ok"] is True
        assert result["enrollment_count"] == 1
        profile_id, vec, model_id, count = mock_services["sqlite_db"].update_speaker_profile.call_args[0]
        assert np.allclose(vec, [0.0, 1.0])
        assert model_id == "new-model"
        assert count == 1

    def test_no_stored_voiceprint(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_speakers.return_value = []

        result = ctrl.enroll_speaker("test", "Speaker 1", "Andrew")
        assert result["ok"] is False
        assert "No voiceprint stored" in result["error"]
        mock_services["sqlite_db"].insert_speaker_profile.assert_not_called()

    def test_empty_person_name_rejected(self, mock_services):
        ctrl = mock_services["controller"]
        result = ctrl.enroll_speaker("test", "Speaker 1", "   ")
        assert result["ok"] is False
        mock_services["sqlite_db"].get_recording_speakers.assert_not_called()

    def test_delete_profile(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].delete_speaker_profile.return_value = True
        assert ctrl.delete_speaker_profile(3)["ok"] is True

        mock_services["sqlite_db"].delete_speaker_profile.return_value = False
        assert ctrl.delete_speaker_profile(3)["ok"] is False

    def test_list_profiles_excludes_embeddings(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_latest_voiceprint_model_id.return_value = None
        mock_services["sqlite_db"].get_speaker_profiles.return_value = [
            {
                "id": 1,
                "name": "Andrew",
                "embedding": np.array([1.0], dtype=np.float32),
                "model_id": "m1",
                "enrollment_count": 2,
                "created_at": "2026-07-07 00:00:00",
            }
        ]

        result = ctrl.list_speaker_profiles()
        assert result["ok"] is True
        assert result["speakers"] == [
            {
                "id": 1,
                "name": "Andrew",
                "model_id": "m1",
                "enrollment_count": 2,
                "created_at": "2026-07-07 00:00:00",
                "stale": False,
            }
        ]

    def test_list_profiles_flags_stale_model(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_latest_voiceprint_model_id.return_value = "new-model"
        mock_services["sqlite_db"].get_speaker_profiles.return_value = [
            {
                "id": 1,
                "name": "Andrew",
                "embedding": np.array([1.0], dtype=np.float32),
                "model_id": "old-model",
                "enrollment_count": 1,
                "created_at": "2026-07-07 00:00:00",
            },
            {
                "id": 2,
                "name": "Zoe",
                "embedding": np.array([1.0], dtype=np.float32),
                "model_id": "new-model",
                "enrollment_count": 1,
                "created_at": "2026-07-07 00:00:00",
            },
        ]

        result = ctrl.list_speaker_profiles()
        assert [s["stale"] for s in result["speakers"]] == [True, False]


class TestDashboardControllerApplySpeakerProfiles:
    def _setup(self, mock_services, transcript, voiceprints, profiles):
        mock_services["whisper_service"].speaker_id_enabled = True
        mock_services["whisper_service"].speaker_id_threshold = 0.5
        mock_services["whisper_service"].speaker_id_margin = 0.05
        mock_services["sqlite_db"].get_speaker_profiles.return_value = profiles
        mock_services["sqlite_db"].get_recordings.return_value = [
            DBRecording(
                id=1, name="rec1", label="R", duration=10,
                created_at=datetime.now(), transcript=transcript,
            )
        ]
        mock_services["sqlite_db"].get_recording_speakers.return_value = voiceprints

    def _profile(self, name, vec, model_id="m"):
        return {
            "id": 1,
            "name": name,
            "embedding": np.asarray(vec, dtype=np.float32),
            "model_id": model_id,
            "enrollment_count": 1,
            "created_at": "2026-07-07 00:00:00",
        }

    def _voiceprint(self, label, vec, model_id="m"):
        return {
            "label": label,
            "embedding": np.asarray(vec, dtype=np.float32),
            "model_id": model_id,
            "speech_seconds": 10.0,
        }

    def test_requires_speaker_id_enabled(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["whisper_service"].speaker_id_enabled = False

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is False
        assert "SPEAKER_ID_ENABLED" in result["error"]

    def test_requires_profiles(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["whisper_service"].speaker_id_enabled = True
        mock_services["sqlite_db"].get_speaker_profiles.return_value = []

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is False
        assert "No voice profiles" in result["error"]

    def test_renames_matched_anonymous_speakers(self, mock_services):
        ctrl = mock_services["controller"]
        self._setup(
            mock_services,
            transcript="[00:00] Speaker 1: Hello\n[00:05] Speaker 2: Hi",
            voiceprints=[
                self._voiceprint("Speaker 1", [1.0, 0.0]),
                self._voiceprint("Speaker 2", [0.0, 1.0]),
            ],
            profiles=[self._profile("Andrew", [1.0, 0.0])],
        )

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is True
        assert result["checked"] == 1
        assert result["updated"] == [{"name": "rec1", "renames": {"Speaker 1": "Andrew"}}]
        mock_services["sqlite_db"].update_transcript.assert_called_once_with(
            "rec1", "[00:00] Andrew: Hello\n[00:05] Speaker 2: Hi"
        )
        mock_services["sqlite_db"].rename_recording_speaker_label.assert_called_once_with(
            "rec1", "Speaker 1", "Andrew"
        )

    def test_skips_when_name_already_in_transcript(self, mock_services):
        # "Speaker 1" matches Andrew's profile, but an "Andrew" speaker already
        # exists in the transcript — renaming would merge two distinct speakers.
        ctrl = mock_services["controller"]
        self._setup(
            mock_services,
            transcript="[00:00] Speaker 1: Hello\n[00:05] Andrew: Hi",
            voiceprints=[self._voiceprint("Speaker 1", [1.0, 0.0])],
            profiles=[self._profile("Andrew", [1.0, 0.0])],
        )

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is True
        assert result["updated"] == []
        mock_services["sqlite_db"].update_transcript.assert_not_called()

    def test_no_match_below_threshold(self, mock_services):
        ctrl = mock_services["controller"]
        self._setup(
            mock_services,
            transcript="[00:00] Speaker 1: Hello",
            voiceprints=[self._voiceprint("Speaker 1", [1.0, 0.0])],
            profiles=[self._profile("Andrew", [0.0, 1.0])],  # cosine 0
        )

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is True
        assert result["checked"] == 1
        assert result["updated"] == []
        mock_services["sqlite_db"].update_transcript.assert_not_called()

    def test_already_named_voiceprints_not_rechecked(self, mock_services):
        # All voiceprints already carry person names: nothing anonymous to fix.
        ctrl = mock_services["controller"]
        self._setup(
            mock_services,
            transcript="[00:00] Andrew: Hello",
            voiceprints=[self._voiceprint("Andrew", [1.0, 0.0])],
            profiles=[self._profile("Andrew", [1.0, 0.0])],
        )

        result = ctrl.apply_speaker_profiles()
        assert result["ok"] is True
        assert result["checked"] == 0
        assert result["updated"] == []


class TestDashboardControllerSummary:
    def test_get_summary_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1,
            name="test",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            summary="# Summary",
            title="Title",
            tags="a,b",
        )

        result = ctrl.get_summary("test")
        assert result["ok"] is True
        assert result["summary"] == "# Summary"
        assert result["title"] == "Title"
        assert result["tags"] == ["a", "b"]

    def test_get_summary_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = None

        result = ctrl.get_summary("test")
        assert result["ok"] is False

    def test_get_summary_no_summary_text(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1, name="test", label="Test", duration=10, created_at=datetime.now(), summary=None
        )

        result = ctrl.get_summary("test")
        assert result["ok"] is False

    def test_get_summary_empty_tags(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1,
            name="test",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            summary="content",
            title="Title",
            tags=None,
        )

        result = ctrl.get_summary("test")
        assert result["ok"] is True
        assert result["tags"] == []


class TestDashboardControllerSummarizeRecording:
    def test_no_transcript(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = None

        result = ctrl.summarize_recording("test", "it/Generale/SintesiAdattiva")
        assert result["ok"] is False
        assert "transcript" in result["error"].lower()

    def test_prompt_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = "transcript text"
        mock_services["system_prompts_repo"].get_prompt_content.return_value = None

        result = ctrl.summarize_recording("test", "invalid/prompt")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_successful_summarization(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = "transcript text"
        mock_services["system_prompts_repo"].get_prompt_content.return_value = "Be concise."
        mock_services["summarization_service"].summarize.return_value = {
            "title": "Result Title",
            "tags": ["tag1", "tag2"],
            "summary": "Result summary",
        }

        result = ctrl.summarize_recording("2026Mar27-094938-Wip01", "it/Generale/SintesiAdattiva")
        assert result["ok"] is True
        assert result["title"] == "Result Title"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["summary"] == "Result summary"
        mock_services["sqlite_db"].save_summarization_result.assert_called_once()

    def test_summarization_failure(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_transcript.return_value = "transcript"
        mock_services["system_prompts_repo"].get_prompt_content.return_value = "prompt"
        mock_services["summarization_service"].summarize.side_effect = Exception("API error")

        result = ctrl.summarize_recording("test", "prompt_id")
        assert result["ok"] is False
        assert "Summarization failed" in result["error"]


class TestDashboardControllerMetadata:
    def test_update_metadata_success(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1, name="test", label="Test", duration=10, created_at=datetime.now()
        )

        result = ctrl.update_recording_metadata("test", "New Title", ["tag1", "tag2"])
        assert result["ok"] is True
        assert result["title"] == "New Title"
        assert result["tags"] == ["tag1", "tag2"]

    def test_update_metadata_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = None

        result = ctrl.update_recording_metadata("ghost", "Title", ["tag"])
        assert result["ok"] is False

    def test_update_metadata_filters_empty_tags(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1, name="test", label="Test", duration=10, created_at=datetime.now()
        )

        result = ctrl.update_recording_metadata("test", "Title", ["good", "", "  ", "also_good"])
        assert result["tags"] == ["good", "also_good"]

    def test_update_summary_content_success(self, mock_services):
        ctrl = mock_services["controller"]
        mock_summary = MagicMock()
        mock_summary.id = 11
        mock_summary.title = "Updated"
        mock_summary.tags = "a,b"
        mock_summary.summary = "Edited content"
        mock_services["sqlite_db"].get_summary_by_id.return_value = mock_summary
        mock_services["sqlite_db"].update_summary_content.return_value = mock_summary

        result = ctrl.update_summary(summary_id=11, summary="Edited content")
        assert result["ok"] is True
        assert result["summary"] == "Edited content"
        mock_services["sqlite_db"].update_summary_content.assert_called_once_with(11, "Edited content")

    def test_update_summary_with_no_fields(self, mock_services):
        ctrl = mock_services["controller"]

        result = ctrl.update_summary(summary_id=11)
        assert result["ok"] is False
        assert "nothing to update" in result["error"].lower()

    def test_update_summary_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["sqlite_db"].get_summary_by_id.return_value = None

        result = ctrl.update_summary(summary_id=999, summary="edited")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()


class TestDashboardControllerAudioPath:
    def test_audio_path_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["local_repo"].get_path.return_value = "/path/to/file.hda"
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1,
            name="test",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            file_extension="hda",
        )

        path, ext = ctrl.get_audio_file_path("test")
        assert path == "/path/to/file.hda"
        assert ext == "hda"

    def test_audio_path_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = False

        path, ext = ctrl.get_audio_file_path("ghost")
        assert path is None
        assert ext == ""

    def test_audio_path_mp3(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].exists.return_value = True
        mock_services["local_repo"].get_path.return_value = "/path/to/file.mp3"
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=2,
            name="test",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            file_extension="mp3",
        )

        path, ext = ctrl.get_audio_file_path("test")
        assert path == "/path/to/file.mp3"
        assert ext == "mp3"


class TestDashboardControllerListPrompts:
    def test_list_system_prompts(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["system_prompts_repo"].get_all.return_value = [
            {"id": "it/Generale/SintesiAdattiva", "label": "Generale / SintesiAdattiva"},
        ]

        result = ctrl.list_system_prompts()
        assert result["ok"] is True
        assert len(result["prompts"]) == 1


class TestDashboardControllerListLocalRecordings:
    def test_list_local(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].get_all.return_value = ["a.hda", "b.hda"]

        result = ctrl.list_local_recordings()
        assert result == ["a.hda", "b.hda"]


class TestDashboardControllerDelete:
    def test_delete_local_and_db(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].delete.return_value = True
        mock_services["sqlite_db"].delete_recording.return_value = True

        result = ctrl.delete_recording("test", delete_local=True, delete_db=True)
        assert result["ok"] is True
        assert "local file" in result["deleted"]
        assert "database record" in result["deleted"]

    def test_delete_local_not_found(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].delete.return_value = False
        mock_services["sqlite_db"].delete_recording.return_value = False

        result = ctrl.delete_recording("test", delete_local=True, delete_db=True)
        assert result["ok"] is False

    def test_delete_partial_success(self, mock_services):
        ctrl = mock_services["controller"]
        mock_services["local_repo"].delete.return_value = True
        mock_services["sqlite_db"].delete_recording.return_value = False

        result = ctrl.delete_recording("test", delete_local=True, delete_db=True)
        assert result["ok"] is True
        assert "local file" in result["deleted"]
        assert len(result["warnings"]) > 0


class TestDashboardControllerPublish:
    def test_get_publish_destinations_empty(self, mock_services):
        ctrl = mock_services["controller"]
        result = ctrl.get_publish_destinations()
        assert result["ok"] is True
        assert result["destinations"] == []

    def test_get_publish_destinations_with_notion(self, mock_services, tmp_path):
        mock_notion = MagicMock()
        mock_notion.is_configured = True

        template_dir = tmp_path / "templates2"
        template_dir.mkdir()
        (template_dir / "home.html").write_text("<html></html>")

        ctrl = DashboardController(
            sqlite_db_repository=mock_services["sqlite_db"],
            local_recordings_repository=mock_services["local_repo"],
            transcription_service=mock_services["transcription_service"],
            system_prompts_repository=mock_services["system_prompts_repo"],
            template_path=str(template_dir),
            publish_services={"notion": mock_notion},
            task_generation_service=mock_services["task_generation_service"],
            summarization_service=MagicMock(),
        )

        result = ctrl.get_publish_destinations()
        assert result["ok"] is True
        assert len(result["destinations"]) == 1
        assert result["destinations"][0]["id"] == "notion"
        assert result["destinations"][0]["label"] == "Notion"

    def test_publish_unknown_destination(self, mock_services):
        ctrl = mock_services["controller"]
        result = ctrl.publish_recording("test", "unknown_dest")
        assert result["ok"] is False
        assert "Unknown" in result["error"]

    def test_publish_no_summary(self, mock_services, tmp_path):
        mock_notion = MagicMock()

        template_dir = tmp_path / "templates3"
        template_dir.mkdir()
        (template_dir / "home.html").write_text("<html></html>")

        ctrl = DashboardController(
            sqlite_db_repository=mock_services["sqlite_db"],
            local_recordings_repository=mock_services["local_repo"],
            transcription_service=mock_services["transcription_service"],
            system_prompts_repository=mock_services["system_prompts_repo"],
            template_path=str(template_dir),
            publish_services={"notion": mock_notion},
            task_generation_service=mock_services["task_generation_service"],
            summarization_service=MagicMock(),
        )
        mock_services["sqlite_db"].get_recording_by_name.return_value = None

        result = ctrl.publish_recording("test", "notion")
        assert result["ok"] is False
        assert "summary" in result["error"].lower()

    def test_publish_success(self, mock_services, tmp_path):
        mock_notion = MagicMock()
        mock_notion.publish_summary.return_value = {"ok": True, "url": "https://notion.so/page"}

        template_dir = tmp_path / "templates4"
        template_dir.mkdir()
        (template_dir / "home.html").write_text("<html></html>")

        ctrl = DashboardController(
            sqlite_db_repository=mock_services["sqlite_db"],
            local_recordings_repository=mock_services["local_repo"],
            transcription_service=mock_services["transcription_service"],
            system_prompts_repository=mock_services["system_prompts_repo"],
            template_path=str(template_dir),
            publish_services={"notion": mock_notion},
            task_generation_service=mock_services["task_generation_service"],
            summarization_service=mock_services["summarization_service"],
        )
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1,
            name="2026Mar27-094938-Wip01",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            summary="# Summary",
            title="My Title",
            tags="a,b",
        )

        result = ctrl.publish_recording("2026Mar27-094938-Wip01", "notion")
        assert result["ok"] is True
        assert result["url"] == "https://notion.so/page"
        mock_services["sqlite_db"].save_notion_url.assert_called_once()

    def test_publish_exception(self, mock_services, tmp_path):
        mock_notion = MagicMock()
        mock_notion.publish_summary.side_effect = Exception("Network error")

        template_dir = tmp_path / "templates5"
        template_dir.mkdir()
        (template_dir / "home.html").write_text("<html></html>")

        ctrl = DashboardController(
            sqlite_db_repository=mock_services["sqlite_db"],
            local_recordings_repository=mock_services["local_repo"],
            transcription_service=mock_services["transcription_service"],
            system_prompts_repository=mock_services["system_prompts_repo"],
            template_path=str(template_dir),
            publish_services={"notion": mock_notion},
            task_generation_service=mock_services["task_generation_service"],
            summarization_service=mock_services["summarization_service"],
        )
        mock_services["sqlite_db"].get_recording_by_name.return_value = DBRecording(
            id=1,
            name="test",
            label="Test",
            duration=10,
            created_at=datetime.now(),
            summary="# Summary",
            title="Title",
            tags="a",
        )

        result = ctrl.publish_recording("test", "notion")
        assert result["ok"] is False
        assert "Publish failed" in result["error"]
