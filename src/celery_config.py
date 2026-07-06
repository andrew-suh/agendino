"""
Celery configuration and initialization
"""
import os
from celery import Celery

# Create Celery app
celery_app = Celery("agendino")

# Task time limits. CPU Whisper on a long recording can legitimately need well over
# 30 minutes — raise CELERY_TASK_TIME_LIMIT for very long audio. The soft limit fires
# SoftTimeLimitExceeded inside the task (clean failure: status update + lock release);
# the hard limit is the kill switch and defaults to 2 minutes above the soft one.
TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", str(60 * 60)))
TASK_SOFT_TIME_LIMIT = int(
    os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", str(max(TASK_TIME_LIMIT - 120, 60)))
)

# Configure Celery with Redis as broker and backend
redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,  # triggers SoftTimeLimitExceeded
    result_expires=3600,  # Results expire after 1 hour
)

# Auto-discover tasks from the celery_tasks module
celery_app.autodiscover_tasks(["celery_tasks"])
