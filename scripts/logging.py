import logging
from scripts.params import LOG_LEVEL, LOG_FILE

def setup_logging(file=LOG_FILE, level=LOG_LEVEL):
    logging.basicConfig(filename=LOG_FILE, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    level = LOG_LEVEL.upper()  
    if level == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == 'INFO':
        logging.getLogger().setLevel(logging.INFO)
    elif level == 'ERROR':
        logging.getLogger().setLevel(logging.ERROR)
    elif level == 'CRITICAL':
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        logging.warning(f'Unknown logging type: {LOG_LEVEL}. Set to default: WARNING.')

