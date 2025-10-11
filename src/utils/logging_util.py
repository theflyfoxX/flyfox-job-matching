import logging

def setup_logging(level=logging.INFO):
    """
    Configures logging for the application.

    Args:
        level (int): Logging level, default is logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)
