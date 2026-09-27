"""Load the challenge TSVs once, cache them as Parquet, and hand out Arrow-backed frames."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from .config import DATA_DIR, WORK_DIR


def _tsv_path(split: str, name: str) -> Path:
    return DATA_DIR / split / f"{split}_{name}.tsv"


def _cache_path(split: str, name: str) -> Path:
    return WORK_DIR / "parquet" / f"{split}_{name}.parquet"


def _read_tsv(path: Path) -> pa.Table:
    with open(path, encoding="utf-8") as f:
        columns = f.readline().rstrip("\n").split("\t")
    # Quoting off: names contain stray quotes and the files are plain tab-separated.
    return pcsv.read_csv(
        path,
        read_options=pcsv.ReadOptions(block_size=1 << 26),
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pcsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
        ),
    )


def load_table(split: str, name: str) -> pa.Table:
    """name is 'source1' | 'source2' | 'source3' | 'ground_truth'."""
    cache = _cache_path(split, name)
    if not cache.exists():
        table = _read_tsv(_tsv_path(split, name))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, cache)
        return table
    return pq.read_table(cache)


def load_source(split: str, source: int) -> pd.DataFrame:
    return load_table(split, f"source{source}").to_pandas(types_mapper=pd.ArrowDtype)


def load_ground_truth() -> pd.DataFrame:
    return load_table("train", "ground_truth").to_pandas(types_mapper=pd.ArrowDtype)


def ground_truth_pairs() -> pd.DataFrame:
    """One row per (source1_entity_id, matched_entity_id) pair; singletons are dropped."""
    import pyarrow.compute as pc

    gt = load_table("train", "ground_truth")
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    flat = pc.list_flatten(lists)
    parents = pc.list_parent_indices(lists)
    s1 = pc.take(gt["source1_entity_id"], parents)
    pairs = pa.table({"s1_id": s1, "match_id": flat})
    pairs = pairs.filter(pc.not_equal(pairs["match_id"], ""))
    return pairs.to_pandas(types_mapper=pd.ArrowDtype)
