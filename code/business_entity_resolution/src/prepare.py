"""Normalize every record once and cache it. Run: python -m src.prepare"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .config import WORK_DIR
from .data import load_table
from .normalize import normalize_address, normalize_name, phonetic
from .translit import has_devanagari

CHUNK = 100_000


def _normalize_chunk(args):
    names, addresses, countries = args
    rows = []
    for name, addr, country in zip(names, addresses, countries):
        core, legal, domain = normalize_name(name)
        a_tokens, a_nums, code = normalize_address(addr, country)
        rows.append((core, legal, domain, phonetic(core), a_tokens, a_nums, code, has_devanagari(name), has_devanagari(addr)))
    return rows


def norm_path(split: str, source: int):
    return WORK_DIR / "norm" / f"{split}_source{source}.parquet"


def prepare(split: str, source: int, workers: int) -> None:
    out = norm_path(split, source)
    if out.exists():
        return
    t0 = time.time()
    df = load_table(split, f"source{source}").to_pandas()
    names, addrs, countries = df["business_name"].tolist(), df["business_address"].tolist(), df["country"].tolist()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK], countries[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(_normalize_chunk, jobs) for r in chunk]
    cols = ["name_core", "name_legal", "name_domain", "name_phon", "addr_norm", "addr_nums", "postcode", "name_indic",
            "addr_indic"]
    norm = pd.DataFrame(rows, columns=cols)
    norm.insert(0, "country", df["country"].values)
    norm.insert(0, "entity_id", df["entity_id"].values)
    out.parent.mkdir(parents=True, exist_ok=True)
    norm.to_parquet(out, index=False)
    print(f"{split} source{source}: {len(norm):,} rows in {time.time() - t0:.0f}s", flush=True)


def load_norm(split: str, source: int) -> pd.DataFrame:
    return pd.read_parquet(norm_path(split, source))


def main() -> None:
    workers = max(1, (os.cpu_count() or 2) - 2)
    for split in ("train", "test"):
        for source in (1, 2, 3):
            prepare(split, source, workers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
