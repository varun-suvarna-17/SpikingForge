"""
Utility helpers.
"""

import logging


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("spikingforge")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger