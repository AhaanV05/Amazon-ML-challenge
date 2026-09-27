"""France-only post-processing: claim unclaimed "suffix variant" records at their S1's exact address.

One generator op writes a copy of a business as "{first word} {legal form} {suffix}" at the business's own address,
e.g. "Pornic Soins SAS" -> "Pornic SAS Groupe", "Entente Federation SARL" -> "Entente SARL & Fils". In India/US this
drop-a-word-add-one shape at the S1's address is 99% true. In France the suffix is drawn evenly from Cie, Services,
Groupe, Developpement, France, & Associes, & Fils / Et Fils (near-identical counts per legal form, 50/50 S2/S3), yet
the models, trained on US text where "Group"/"Holdings" mark hard negatives, claim 97% of the Cie/Services copies and
only 9-35% of the Groupe/Developpement/Fils/France ones. This adds those records when nobody claims them and exactly
one S1 sits at the record's address. Every other row of the base submission is kept.

    python -m src.france_suffix SCORES BASE_DIR OUT_DIR
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .candidates import d_key, d_str, q_key
from .config import WORK_DIR
from .normalize import FRANCE_REGIONS
from .redecode import _read_lists
from .submission import write_submission

SUFFIX = {"groupe", "developpement", "fils", "france", "associes", "cie", "services", "service", "frs"}
NAME_STOP = {"and", "de", "la", "le", "les", "du", "des", "d", "l", "the", "of", "et"}
ADDR_STOP = set(FRANCE_REGIONS.values()) | {"bis", "b", "ter", "t", "no", "n", "apt", "appartement"}
NUM = re.compile(r"^\d+[a-z]?$")


def _records(source: int) -> pd.DataFrame:
    df = pq.read_table(WORK_DIR / "norm" / f"test_source{source}.parquet", filters=[("country", "==", "France")],
                       columns=["entity_id", "name_core", "addr_norm", "addr_nums"]).to_pandas()
    df.index = q_key(df["entity_id"]) if source == 1 else d_key(df["entity_id"])
    return pd.DataFrame({
        "name": df["name_core"].fillna("").str.split().map(lambda t: [x for x in t if x not in NAME_STOP]),
        "nums": df["addr_nums"].fillna("").str.split().map(lambda t: frozenset(x.lstrip("0") for x in t)),
        "street": df["addr_norm"].fillna("").str.split().map(
            lambda t: frozenset(x for x in t if not NUM.match(x) and x not in ADDR_STOP and len(x) > 1)),
    }, index=df.index)


def _same_address(q: pd.DataFrame, d: pd.DataFrame) -> pd.Series:
    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0
    return pd.Series([bool(a) and a == b and jac(s, t) >= 0.5
                      for a, b, s, t in zip(q["nums"], d["nums"], q["street"], d["street"])])


def _suffix_copy(q_name: list[str], d_name: list[str]) -> bool:
    """Record keeps the S1's first name word, drops at least one S1 word and adds exactly one word, a suffix."""
    if not q_name or q_name[0] not in d_name:
        return False
    added, dropped = set(d_name) - set(q_name), set(q_name) - set(d_name)
    return len(added) == 1 and bool(dropped) and next(iter(added)) in SUFFIX


def additions(scores: str, matches: dict[str, list[str]]) -> pd.DataFrame:
    te = pq.read_table(scores, filters=[("country", "==", "France")], columns=["q_id", "d_id"]).to_pandas()
    s1, docs = _records(1), pd.concat([_records(2), _records(3)])
    q, d = s1.loc[te["q_id"]].reset_index(drop=True), docs.loc[te["d_id"]].reset_index(drop=True)
    te = te.reset_index(drop=True).assign(at_addr=_same_address(q, d))
    te["n_at_addr"] = te.groupby("d_id")["at_addr"].transform("sum")
    claimed = {x for ids in matches.values() for x in ids}
    te["claimed"] = pd.Series(d_str(te["d_id"])).isin(claimed)
    keep = te["at_addr"] & (te["n_at_addr"] == 1) & ~te["claimed"]
    shape = pd.Series([_suffix_copy(a, b) for a, b in zip(q["name"], d["name"])])
    return te.loc[keep & shape, ["q_id", "d_id"]]


def main(scores: str, base_dir: str, out_dir: str) -> None:
    matches = _read_lists(Path(base_dir) / "matching_results.tsv")
    candidates = _read_lists(Path(base_dir) / "candidate_pairs.tsv")
    add = additions(scores, matches)
    for q_id, d_id in zip(add["q_id"], add["d_id"]):
        matches[f"S1-{q_id}"].append(d_str([d_id])[0])
    write_submission(Path(out_dir), list(candidates), matches, candidates)
    print(f"France suffix copies: added {len(add):,} records to {add['q_id'].nunique():,} entities")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
