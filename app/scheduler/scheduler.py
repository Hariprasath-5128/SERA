import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from app import config
from app.scheduler.jobs.pattern_finder import scan_and_trigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

def _on_job_event(event):
    if event.exception:
        logger.error(f"Scheduler job {event.job_id} failed with exception: {event.exception}")
    else:
        logger.debug(f"Scheduler job {event.job_id} executed successfully")

def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(
        job_defaults={
            "misfire_grace_time": 30,   # seconds before job is abandoned if late
            "coalesce": True,           # merge missed ticks into one
            "max_instances": 1,         # CRITICAL: no overlapping pattern_finder runs
        }
    )
    
    scheduler.add_job(
        func=scan_and_trigger,
        trigger=IntervalTrigger(minutes=config.POLL_INTERVAL_MINUTES),
        id="pattern_finder",
        replace_existing=True,
    )
    
    scheduler.add_listener(_on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    
    return scheduler

def get_scheduler() -> BackgroundScheduler | None:
    """Returns the running scheduler singleton (used by admin status endpoint)."""
    return _scheduler
