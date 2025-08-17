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
