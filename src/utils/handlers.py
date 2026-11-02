"""Logging handlers for the Hydra job logging (`configs/hydra/default.yaml`)."""

import logging
from pathlib import Path


class MkdirFileHandler(logging.FileHandler):
    """`logging.FileHandler` that creates the folder of the log file (Hydra only creates the run dir, not `logs/` in it)."""

    def __init__(self, filename, mode="a", encoding=None, delay=False, errors=None):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode, encoding, delay, errors)
