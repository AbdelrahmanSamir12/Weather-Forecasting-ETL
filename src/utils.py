import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

def log_and_raise(msg, exc):
    logging.error(msg)
    raise exc(msg)