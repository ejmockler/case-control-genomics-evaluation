import logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if they already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)