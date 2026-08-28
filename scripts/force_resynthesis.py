import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath('.'))
from app.scheduler.jobs.pattern_finder import scan_and_trigger

if __name__ == '__main__':
    logger.info('Starting forced re-synthesis via Groq API key...')
    scan_and_trigger()
    logger.info('Re-synthesis complete!')
