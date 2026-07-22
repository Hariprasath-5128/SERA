import logging
from app.maintenance.staleness_checker import run_soft_staleness_check
from app.maintenance.decay_scorer import compute_and_apply_decay
from app.maintenance.hierarchy_merger import merge_similar_nodes

logger = logging.getLogger(__name__)

def run_staleness_job() -> None:
    logger.info("Starting scheduled maintenance job: staleness check...")
    try:
        run_soft_staleness_check()
        logger.info("Finished scheduled maintenance job: staleness check.")
    except Exception as e:
        logger.error(f"Scheduled maintenance job failed: staleness check: {e}", exc_info=True)


def run_decay_job() -> None:
    logger.info("Starting scheduled maintenance job: decay scorer...")
    try:
        compute_and_apply_decay()
        logger.info("Finished scheduled maintenance job: decay scorer.")
    except Exception as e:
        logger.error(f"Scheduled maintenance job failed: decay scorer: {e}", exc_info=True)


def run_merger_job() -> None:
    logger.info("Starting scheduled maintenance job: hierarchy merger...")
    try:
        merge_similar_nodes()
        logger.info("Finished scheduled maintenance job: hierarchy merger.")
    except Exception as e:
        logger.error(f"Scheduled maintenance job failed: hierarchy merger: {e}", exc_info=True)
