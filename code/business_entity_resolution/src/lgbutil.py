import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import THREADS

CHUNK = 4_000_000


def predict_chunked(model: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    """Predict in float32 slices so a 30M-row frame never becomes one float64 matrix."""
    return predict_frames(model, [frame])


def predict_frames(model: lgb.Booster, frames: list[pd.DataFrame]) -> np.ndarray:
    """Predict from several row-aligned frames without ever concatenating them in full."""
    cols = model.feature_name()
    n = len(frames[0])
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, CHUNK):
        part = pd.concat([f.iloc[i:i + CHUNK].reset_index(drop=True) for f in frames], axis=1)
        out[i:i + CHUNK] = model.predict(part[cols].to_numpy(np.float32), num_threads=THREADS)
    return out
