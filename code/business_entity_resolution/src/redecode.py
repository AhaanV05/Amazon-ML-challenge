"""Re-decode saved test pair probabilities (test_scores.parquet, written by step_predict) for some countries, keeping
the other countries' rows of a base submission. Used to change decoding for France alone, whose data we cannot
validate on.

    python -m src.redecode SCORES BASE_DIR OUT_DIR COUNTRY [COUNTRY ...]
"""
import sys
from pathlib import Path

import pandas as pd

from .candidates import d_str
from .decode import decode_expected_f, one_owner
from .submission import write_submission


def _read_lists(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s1: [x for x in ids.split(",") if x] for s1, ids in zip(df.iloc[:, 0], df.iloc[:, 1])}


def main(scores: str, base_dir: str, out_dir: str, countries: list[str]) -> None:
    te = pd.read_parquet(scores)
    te = te[te["country"].isin(countries)]
    matches = _read_lists(Path(base_dir) / "matching_results.tsv")
    candidates = _read_lists(Path(base_dir) / "candidate_pairs.tsv")
    for q in te["q_id"].unique():
        matches[f"S1-{q}"] = []
    matches.update({f"S1-{q}": d_str(ds) for q, ds in decode_expected_f(one_owner(te)).items()})
    write_submission(Path(out_dir), list(candidates), matches, candidates)
    n = te["q_id"].nunique()
    print(f"re-decoded {n:,} entities ({', '.join(countries)}); matches/entity "
          f"{sum(len(matches[f'S1-{q}']) for q in te['q_id'].unique()) / n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
