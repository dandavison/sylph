import os
from pathlib import Path


def get_download_directory():
    dir = os.getenv("XENO_QUERO_DOWNLOAD_DIRECTORY", "").strip()
    if not dir:
        raise AssertionError("XENO_QUERO_DOWNLOAD_DIRECTORY env var is not set.")
    dir = Path(dir)
    if not dir.is_dir():
        raise AssertionError(f"{dir} is not a directory.")
    return dir
