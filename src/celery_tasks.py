"""
Celery tasks for long-running operations like transcription and summarization
"""
import logging

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

import task_locks
from app.depends import get_dashboard_controller, get_sqlite_db_repository
from celery_config import celery_app, TASK_SOFT_TIME_LIMIT
from services.WhisperTranscriptionService import DiarizationSetupError

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task class that builds the DashboardController and cleans up task locks."""
    autoretry_for = (Exception,)
    # Setup errors (gated HF model terms not accepted, missing token) and soft
    # time-limit kills won't fix themselves on retry — fail immediately instead
    # of re-running transcription 3x (a timed-out job just times out again).
    dont_autoretry_for = (DiarizationSetupError, SoftTimeLimitExceeded)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True

    def get_dashboard_controller(self):
        """Build a DashboardController using the same dependency wiring as the API."""
        return get_dashboard_controller()

    def _lock_key_for(self, args, kwargs) -> str | None:
        """Reconstruct the Redis lock key this task holds from its name + arguments."""
        def arg(pos, key):
            if args and len(args) > pos:
                return args[pos]
            return kwargs.get(key)

        if self.name == "celery_tasks.transcribe_audio":
            name = arg(0, "recording_name")
            return task_locks.transcribe_lock_key(name) if name is not None else None
        if self.name == "celery_tasks.summarize_audio":
            name = arg(0, "recording_name")
            prompt_id = arg(1, "prompt_id")
            return task_locks.summarize_lock_key(name, prompt_id) if name is not None else None
        if self.name == "celery_tasks.generate_tasks":
            summary_id = arg(0, "summary_id")
            return task_locks.generate_lock_key(summary_id) if summary_id is not None else None
        return None

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Clean up on terminal returns (success or retry-exhausted failure).

        Runs after EVERY return, including RETRY — skip those, or the lock is
        released while the task is still re-running under the same task_id and
        /tasks/active loses track of it.
        """
        if status == "RETRY":
            return
        key = self._lock_key_for(args, kwargs)
        if key:
            task_locks.release(key)
        # Reflect a terminal failure in the recording's transcription status.
        if status == "FAILURE" and self.name == "celery_tasks.transcribe_audio":
            name = args[0] if args else kwargs.get("recording_name")
            if name is not None:
                try:
                    get_sqlite_db_repository().set_transcription_status(name, "failed")
                except Exception as e:
                    logger.warning(f"Failed to mark transcription failed for {name}: {e}")


@celery_app.task(base=DatabaseTask, bind=True, name="celery_tasks.transcribe_audio")
def transcribe_audio_task(self, recording_name: str, engine: str = "gemini"):
    """
    Transcribe an audio recording using the specified engine

    Args:
        recording_name: Name of the recording to transcribe
        engine: Transcription engine ('gemini' or 'whisper')

    Returns:
        dict with status information
    """
    try:
        logger.info(f"Starting transcription for {recording_name} with {engine} engine")
        self.update_state(state="PROGRESS", meta={"status": "transcribing"})

        controller = self.get_dashboard_controller()
        controller.set_transcription_status(recording_name, "running")
        result = controller.transcribe_recording(recording_name, engine=engine)

        if result.get("ok"):
            logger.info(f"Transcription completed for {recording_name}")
            controller.set_transcription_status(recording_name, "done")
            return {
                "ok": True,
                "transcript": result.get("transcript"),
                "cached": result.get("cached", False),
            }
        else:
            logger.error(f"Transcription failed for {recording_name}: {result.get('error')}")
            raise Exception(result.get("error", "Transcription failed"))

    except SoftTimeLimitExceeded:
        # Re-raise with the same type (skips autoretry) but a message the UI can show.
        logger.error(f"Transcription of {recording_name} hit the Celery soft time limit")
        raise SoftTimeLimitExceeded(
            f"Transcription timed out after {TASK_SOFT_TIME_LIMIT}s — for long recordings, "
            "raise CELERY_TASK_TIME_LIMIT (see docs/celery-guide.md)"
        )
    except Exception as e:
        logger.error(f"Task error during transcription of {recording_name}: {str(e)}")
        raise


@celery_app.task(base=DatabaseTask, bind=True, name="celery_tasks.summarize_audio")
def summarize_audio_task(self, recording_name: str, prompt_id: str):
    """
    Summarize an audio recording using the specified prompt

    Args:
        recording_name: Name of the recording to summarize
        prompt_id: ID of the prompt to use for summarization

    Returns:
        dict with status information
    """
    try:
        logger.info(f"Starting summarization for {recording_name} with prompt {prompt_id}")
        self.update_state(state="PROGRESS", meta={"status": "summarizing"})

        controller = self.get_dashboard_controller()
        result = controller.summarize_recording(recording_name, prompt_id)

        if result.get("ok"):
            logger.info(f"Summarization completed for {recording_name}")
            return {
                "ok": True,
                "summary_id": result.get("summary_id"),
                "message": result.get("message", "Summarization completed"),
            }
        else:
            logger.error(f"Summarization failed for {recording_name}: {result.get('error')}")
            raise Exception(result.get("error", "Summarization failed"))

    except Exception as e:
        logger.error(f"Task error during summarization of {recording_name}: {str(e)}")
        raise


@celery_app.task(base=DatabaseTask, bind=True, name="celery_tasks.generate_tasks")
def generate_tasks_task(self, summary_id: int):
    """
    Generate tasks from a summary

    Args:
        summary_id: ID of the summary to generate tasks from

    Returns:
        dict with status information
    """
    try:
        logger.info(f"Starting task generation for summary {summary_id}")
        self.update_state(state="PROGRESS", meta={"status": "generating_tasks"})

        controller = self.get_dashboard_controller()
        result = controller.generate_tasks(summary_id)

        if result.get("ok"):
            logger.info(f"Task generation completed for summary {summary_id}")
            return {
                "ok": True,
                "tasks_count": result.get("tasks_count", 0),
                "message": result.get("message", "Tasks generated"),
            }
        else:
            logger.error(f"Task generation failed for summary {summary_id}: {result.get('error')}")
            raise Exception(result.get("error", "Task generation failed"))

    except Exception as e:
        logger.error(f"Task error during task generation for summary {summary_id}: {str(e)}")
        raise
