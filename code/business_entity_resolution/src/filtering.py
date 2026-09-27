"""Apply the trained candidate filter to stored retrieval candidates, one country/source file at a time."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidates import cand_dir
from .lgbutil import predict_chunked
from .train import FILTER, MODEL_DIR, apply_filter, filter_features


def filter_candidates(split: str, tag: str) -> pd.DataFrame:
    t0 = time.time()
    filt = lgb.Booster(model_file=str(MODEL_DIR / "filter.txt"))
    parts = []
    for path in sorted(cand_dir(split, tag).glob("*.parquet")):
        cand = pd.read_parquet(path)
        if cand.empty:
            continue
        country, source = path.stem.rsplit("_s", 1)
        cand = cand.assign(country=country, source=np.int8(source))
        kept = apply_filter(cand, predict_chunked(filt, filter_features(cand)), **FILTER)
        parts.append(kept)
        print(f"  filter {split}/{tag}/{path.stem}: {len(cand):,} -> {len(kept):,} | {time.time() - t0:.0f}s", flush=True)
        del cand
    kept = pd.concat(parts, ignore_index=True)
    kept["country"] = kept["country"].astype("category")
    return kept
