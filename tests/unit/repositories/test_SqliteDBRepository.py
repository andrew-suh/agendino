import os
import uuid
from datetime import datetime
from random import randint

import numpy as np
import pytest

from models.DBRecording import DBRecording
from repositories.SqliteDBRepository import SqliteDBRepository


class TestSqliteDBRepository:

    @pytest.fixture
    def init_path(self):
        path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(path, "../../../settings/db_init.sql")

    @pytest.fixture
    def db(self, tmp_path, init_path):
        return SqliteDBRepository("test_unit.db", str(tmp_path), init_path)

    @pytest.fixture
    def sample_recording(self):
        return DBRecording(
            id=None,
            name=f"2026Mar27-094938-Wip{randint(1, 99):02d}",
            label="Test Recording",
            duration=120,
            created_at=datetime(2026, 3, 27, 9, 49, 38),
        )

    def test_insert_and_get_recordings(self, db, sample_recording):
        row_id = db.insert_recording(sample_recording)
        assert row_id is not None

        recordings = db.get_recordings()
        assert len(recordings) == 1
        assert recordings[0].name == sample_recording.name
        assert recordings[0].label == sample_recording.label
        assert recordings[0].duration == sample_recording.duration

    def test_get_recording_by_name(self, db, sample_recording):
        db.insert_recording(sample_recording)

        found = db.get_recording_by_name(sample_recording.name)
        assert found is not None
        assert found.name == sample_recording.name

    def test_get_recording_by_name_not_found(self, db):
        result = db.get_recording_by_name("nonexistent")
        assert result is None

    def test_save_and_get_transcript(self, db, sample_recording):
        db.insert_recording(sample_recording)
        assert db.get_transcript(sample_recording.name) is None

        db.save_transcript(sample_recording.name, "Hello, this is a transcript.")
        transcript = db.get_transcript(sample_recording.name)
        assert transcript == "Hello, this is a transcript."

    def test_update_transcript(self, db, sample_recording):
        db.insert_recording(sample_recording)
        db.save_transcript(sample_recording.name, "Before")

        updated = db.update_transcript(sample_recording.name, "After")
        assert updated is True
        assert db.get_transcript(sample_recording.name) == "After"

    def test_update_transcript_not_found(self, db):
        updated = db.update_transcript("ghost", "text")
        assert updated is False

    def test_get_transcript_not_found(self, db):
        result = db.get_transcript("nonexistent")
        assert result is None

    def test_save_and_get_recording_speakers(self, db, sample_recording):
        db.insert_recording(sample_recording)
        speakers = [
            {
                "label": "Speaker 1",
                "embedding": np.array([0.6, 0.8], dtype=np.float32),
                "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "speech_seconds": 12.5,
            },
            {
                "label": "Speaker 2",
                "embedding": [0.0, 1.0],  # plain lists must round-trip too
                "model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
                "speech_seconds": 7.0,
            },
        ]
        db.save_recording_speakers(sample_recording.name, speakers)

        stored = db.get_recording_speakers(sample_recording.name)
        assert [s["label"] for s in stored] == ["Speaker 1", "Speaker 2"]
        assert np.allclose(stored[0]["embedding"], [0.6, 0.8])
        assert stored[0]["embedding"].dtype == np.float32
        assert stored[0]["model_id"] == "pyannote/wespeaker-voxceleb-resnet34-LM"
        assert stored[0]["speech_seconds"] == 12.5

    def test_save_recording_speakers_replaces_previous(self, db, sample_recording):
        db.insert_recording(sample_recording)
        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [1.0], "model_id": "old"}],
        )
        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [2.0], "model_id": "new"}],
        )
        stored = db.get_recording_speakers(sample_recording.name)
        assert len(stored) == 1
        assert stored[0]["model_id"] == "new"

    def test_save_recording_speakers_unknown_recording_is_noop(self, db):
        db.save_recording_speakers(
            "ghost", [{"label": "Speaker 1", "embedding": [1.0], "model_id": "m"}]
        )
        assert db.get_recording_speakers("ghost") == []

    def test_clear_transcript_removes_voiceprints(self, db, sample_recording):
        db.insert_recording(sample_recording)
        db.save_transcript(sample_recording.name, "text")
        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [1.0], "model_id": "m"}],
        )

        assert db.clear_transcript(sample_recording.name) is True
        assert db.get_recording_speakers(sample_recording.name) == []

    def test_rename_recording_speaker_label(self, db, sample_recording):
        db.insert_recording(sample_recording)
        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [1.0], "model_id": "m"}],
        )

        db.rename_recording_speaker_label(sample_recording.name, "Speaker 1", "Andrew")
        assert [s["label"] for s in db.get_recording_speakers(sample_recording.name)] == ["Andrew"]

    def test_get_latest_voiceprint_model_id(self, db, sample_recording):
        assert db.get_latest_voiceprint_model_id() is None

        db.insert_recording(sample_recording)
        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [1.0], "model_id": "old-model"}],
        )
        assert db.get_latest_voiceprint_model_id() == "old-model"

        db.save_recording_speakers(
            sample_recording.name,
            [{"label": "Speaker 1", "embedding": [1.0], "model_id": "new-model"}],
        )
        assert db.get_latest_voiceprint_model_id() == "new-model"

    def test_speaker_profile_crud(self, db):
        profile_id = db.insert_speaker_profile("Andrew", np.array([0.6, 0.8], dtype=np.float32), "m1")
        assert profile_id is not None

        profile = db.get_speaker_profile_by_name("Andrew")
        assert profile["id"] == profile_id
        assert np.allclose(profile["embedding"], [0.6, 0.8])
        assert profile["model_id"] == "m1"
        assert profile["enrollment_count"] == 1

        db.update_speaker_profile(profile_id, [0.0, 1.0], "m2", 3)
        updated = db.get_speaker_profile_by_name("Andrew")
        assert np.allclose(updated["embedding"], [0.0, 1.0])
        assert updated["model_id"] == "m2"
        assert updated["enrollment_count"] == 3

        assert db.delete_speaker_profile(profile_id) is True
        assert db.get_speaker_profile_by_name("Andrew") is None
        assert db.delete_speaker_profile(profile_id) is False

    def test_get_speaker_profiles_sorted_by_name(self, db):
        db.insert_speaker_profile("Zoe", [1.0], "m")
        db.insert_speaker_profile("Andrew", [1.0], "m")
        assert [p["name"] for p in db.get_speaker_profiles()] == ["Andrew", "Zoe"]

    def test_speaker_profile_name_is_unique(self, db):
        import sqlite3

        db.insert_speaker_profile("Andrew", [1.0], "m")
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_speaker_profile("Andrew", [2.0], "m")

    def test_save_and_get_summary(self, db, sample_recording):
        db.insert_recording(sample_recording)
        assert db.get_summary(sample_recording.name) is None

        db.save_summary(sample_recording.name, "# Summary\nThis is a test.")
        summary = db.get_summary(sample_recording.name)
        assert summary == "# Summary\nThis is a test."

    def test_get_summary_not_found(self, db):
        result = db.get_summary("nonexistent")
        assert result is None

    def test_save_summarization_result(self, db, sample_recording):
        db.insert_recording(sample_recording)

        db.save_summarization_result(
            sample_recording.name,
            summary="Full summary text",
            title="Meeting Notes",
            tags="meeting,notes,work",
        )

        rec = db.get_recording_by_name(sample_recording.name)
        assert rec.summary == "Full summary text"
        assert rec.title == "Meeting Notes"
        assert rec.tags == "meeting,notes,work"
        # label should also be updated to the title
        assert rec.label == "Meeting Notes"

    def test_update_title_and_tags(self, db, sample_recording):
        db.insert_recording(sample_recording)

        db.update_title_and_tags(sample_recording.name, "New Title", "tag1,tag2")

        rec = db.get_recording_by_name(sample_recording.name)
        assert rec.title == "New Title"
        assert rec.tags == "tag1,tag2"
        assert rec.label == "New Title"

    def test_update_summary_content(self, db, sample_recording):
        db.insert_recording(sample_recording)
        saved = db.save_summarization_result(
            sample_recording.name,
            summary="Old summary",
            title="Meeting Notes",
            tags="meeting,notes,work",
        )

        updated = db.update_summary_content(saved.id, "New summary")
        assert updated is not None
        assert updated.summary == "New summary"

    def test_update_summary_content_not_found(self, db):
        updated = db.update_summary_content(999999, "New summary")
        assert updated is None

    def test_delete_recording(self, db, sample_recording):
        db.insert_recording(sample_recording)
        assert db.get_recording_by_name(sample_recording.name) is not None

        result = db.delete_recording(sample_recording.name)
        assert result is True
        assert db.get_recording_by_name(sample_recording.name) is None

    def test_delete_recording_not_found(self, db):
        result = db.delete_recording("nonexistent")
        assert result is False

    def test_save_notion_url(self, db, sample_recording):
        db.insert_recording(sample_recording)

        db.save_notion_url(sample_recording.name, "https://notion.so/page123")

        rec = db.get_recording_by_name(sample_recording.name)
        assert rec.notion_url == "https://notion.so/page123"

    def test_multiple_recordings(self, db):
        for i in range(5):
            rec = DBRecording(
                id=None,
                name=f"rec_{i}",
                label=f"Recording {i}",
                duration=i * 60,
                created_at=datetime(2026, 4, 1),
            )
            db.insert_recording(rec)

        all_recs = db.get_recordings()
        assert len(all_recs) == 5

    def test_migration_on_existing_db(self, tmp_path, init_path):
        """Creating a repo twice on the same db should trigger migration path without error."""
        db1 = SqliteDBRepository("migrate_test.db", str(tmp_path), init_path)
        rec = DBRecording(id=None, name="test", label="Test", duration=10, created_at=datetime.now())
        db1.insert_recording(rec)

        # Second instantiation triggers _migrate_db instead of _initialize_db
        db2 = SqliteDBRepository("migrate_test.db", str(tmp_path), init_path)
        recordings = db2.get_recordings()
        assert len(recordings) == 1
