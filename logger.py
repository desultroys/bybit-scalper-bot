"""
logger.py — Console + dosya log sistemi
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from config import CFG


def setup_logger(name: str = "scalper") -> logging.Logger:
    """Renkli console + rotating file logger kur."""
    os.makedirs(CFG.log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ─── Renk kodları (ANSI) ──────────────────────────────────────
    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
        "RESET":    "\033[0m",
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            color = COLORS.get(record.levelname, COLORS["RESET"])
            reset = COLORS["RESET"]
            record.levelname = f"{color}{record.levelname:<8}{reset}"
            return super().format(record)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    # Dosya handler (10MB, 5 backup)
    fh = RotatingFileHandler(
        os.path.join(CFG.log_dir, "bot.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = setup_logger()
