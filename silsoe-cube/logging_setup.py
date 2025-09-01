import logging
import os
from logging.handlers import RotatingFileHandler

# def setup_console_logging(level=logging.INFO):
#     root = logging.getLogger()
#     if root.handlers:
#         return
#     root.setLevel(level)
#     ch = logging.StreamHandler()
#     ch.setLevel(level)
#     fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
#     ch.setFormatter(fmt)
#     root.addHandler(ch)

# def attach_file_logger(outdir, filename="simulation.log", level=logging.DEBUG, max_bytes=5_000_000, backup_count=3):
#     os.makedirs(outdir, exist_ok=True)
#     log_path = os.path.join(outdir, filename)
#     fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
#     fh.setLevel(level)
#     fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s [%(process)d]: %(message)s", "%Y-%m-%d %H:%M:%S")
#     fh.setFormatter(fmt)
#     logging.getLogger().addHandler(fh)
#     logging.getLogger(__name__).debug("Attached file logger: %s", log_path)

def setup_console_logging(level=logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    root.addHandler(ch)

def attach_file_logger(outdir: str, filename: str = "simulation.log", level=logging.DEBUG):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, filename)
    root = logging.getLogger()
    # avoid duplicate file handlers for same file
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path):
            return
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s [%(process)d]: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logging.getLogger(__name__).debug("Attached file logger: %s", log_path)
