import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", PROJECT_ROOT / "Datasets" / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", PROJECT_ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", PROJECT_ROOT / "output"))

THREADS = int(os.environ.get("BER_THREADS", max(1, (os.cpu_count() or 4) - 2)))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SEED = 42
