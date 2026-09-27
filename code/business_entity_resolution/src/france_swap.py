"""France-only post-processing, second rule: claim unclaimed records at their S1's exact address whose name is the S1's
name plus added words ("Union du Bring" -> "Union du Bring Développement") or with one word misspelled ("VPJ Residence"
-> "VPAJ Residence").

In India/US validation, near-copies at the S1's exact address with no other S1 there are true matches 90-94% of the
time, and in France the matcher accepts them at rates that depend on the word itself, the bias the suffix rule
(france_suffix.py) corrected. Swaps of one real word for another ("Institut" -> "Maternelle") are left alone unless
--swaps is given: in France they may be a second business at the same address. Same conditions as the suffix rule:
the record is in the S1's candidate list, nobody claimed it, exactly one S1 sits at its address, and the whole
address agrees closely.

    python -m src.france_swap SCORES BASE_DIR OUT_DIR [--swaps]
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from .candidates import d_key, d_str, q_key
from .config import WORK_DIR
from .france_suffix import _records, _same_address
from .redecode import _read_lists
from .submission import write_submission

ADDR_MIN = 85  # token-set similarity of the whole normalized addresses


def _addresses(source: int) -> pd.Series:
    df = pq.read_table(WORK_DIR / "norm" / f"test_source{source}.parquet", filters=[("country", "==", "France")],
                       columns=["entity_id", "addr_norm"]).to_pandas()
    return pd.Series(df["addr_norm"].fillna("").to_numpy(),
                     index=q_key(df["entity_id"]) if source == 1 else d_key(df["entity_id"]))


def _near_copy(q_name: list[str], d_name: list[str], swaps: bool) -> bool:
    """All S1 words kept and at least one added, or exactly one word replaced by a near-spelling of it (any word, with
    swaps=True, as long as another S1 word is kept)."""
    a, b = set(q_name), set(d_name)
    added, dropped = b - a, a - b
    if not a or not added:
        return False
    if not dropped:
        return True
    if len(dropped) != 1 or len(added) != 1:
        return False
    return (swaps and bool(a & b)) or fuzz.ratio(next(iter(dropped)), next(iter(added))) >= 70


def additions(scores: str, matches: dict[str, list[str]], swaps: bool = False) -> pd.DataFrame:
    te = pq.read_table(scores, filters=[("country", "==", "France")], columns=["q_id", "d_id"]).to_pandas()
    s1, docs = _records(1), pd.concat([_records(2), _records(3)])
    q, d = s1.loc[te["q_id"]].reset_index(drop=True), docs.loc[te["d_id"]].reset_index(drop=True)
    te = te.reset_index(drop=True).assign(at_addr=_same_address(q, d))
    te["n_at_addr"] = te.groupby("d_id")["at_addr"].transform("sum")
    claimed = {x for ids in matches.values() for x in ids}
    te["claimed"] = pd.Series(d_str(te["d_id"])).isin(claimed)
    keep = te["at_addr"] & (te["n_at_addr"] == 1) & ~te["claimed"]
    shape = pd.Series([_near_copy(a, b, swaps) for a, b in zip(q["name"], d["name"])])
    te = te[keep & shape]
    qa = _addresses(1).reindex(te["q_id"].to_numpy()).tolist()
    da = pd.concat([_addresses(2), _addresses(3)]).reindex(te["d_id"].to_numpy()).tolist()
    close = process.cpdist(qa, da, scorer=fuzz.token_set_ratio, workers=-1) >= ADDR_MIN
    return te.loc[close, ["q_id", "d_id"]]


def main(scores: str, base_dir: str, out_dir: str, swaps: bool = False) -> None:
    matches = _read_lists(Path(base_dir) / "matching_results.tsv")
    candidates = _read_lists(Path(base_dir) / "candidate_pairs.tsv")
    add = additions(scores, matches, swaps)
    for q_id, d_id in zip(add["q_id"], add["d_id"]):
        matches[f"S1-{q_id}"].append(d_str([d_id])[0])
    write_submission(Path(out_dir), list(candidates), matches, candidates)
    print(f"France near copies: added {len(add):,} records to {add['q_id'].nunique():,} entities")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--swaps" in sys.argv[4:])
