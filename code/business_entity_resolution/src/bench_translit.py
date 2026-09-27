"""Compare transliterators on real true pairs. Run: python -m src.bench_translit"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from anyascii import anyascii
from indic_transliteration import sanscript
from rapidfuzz import fuzz, process

from .config import SEED
from .data import ground_truth_pairs, load_table
from .translit import transliterate

BLOCKS = {
    "Devanagari": (0x0900, sanscript.DEVANAGARI), "Bengali": (0x0980, sanscript.BENGALI),
    "Gurmukhi": (0x0A00, sanscript.GURMUKHI), "Gujarati": (0x0A80, sanscript.GUJARATI),
    "Oriya": (0x0B00, sanscript.ORIYA), "Tamil": (0x0B80, sanscript.TAMIL), "Telugu": (0x0C00, sanscript.TELUGU),
    "Kannada": (0x0C80, sanscript.KANNADA), "Malayalam": (0x0D00, sanscript.MALAYALAM),
}
_RUN = re.compile(r"[ऀ-ൿ]+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _script_of(run: str) -> str:
    for name, (start, _) in BLOCKS.items():
        if start <= ord(run[0]) < start + 0x80:
            return name
    return "Devanagari"


def _sanscript(s: str, target: str) -> str:
    return _RUN.sub(lambda m: sanscript.transliterate(m.group(0), BLOCKS[_script_of(m.group(0))][1], target), s)


def clean(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))
    return " ".join(_NON_ALNUM.sub(" ", s).split())


METHODS = {
    "ours": transliterate,
    "sanscript_iast": lambda s: _sanscript(s, sanscript.IAST),
    "sanscript_optitrans": lambda s: _sanscript(s, sanscript.OPTITRANS),
    "anyascii": anyascii,
}


def main(per_script: int = 1500) -> None:
    rng = np.random.default_rng(SEED)
    samples = []
    for s in (2, 3):
        t = load_table("train", f"source{s}").select(["entity_id", "business_name"])
        for name, (start, _) in BLOCKS.items():
            mask = pc.match_substring_regex(t["business_name"], f"[{chr(start)}-{chr(start + 0x7f)}]")
            sub = t.filter(mask)
            if sub.num_rows:
                idx = rng.choice(sub.num_rows, size=min(per_script // 2, sub.num_rows), replace=False)
                samples.append(sub.take(pa.array(idx)).to_pandas().assign(script=name))
    sample = pd.concat(samples, ignore_index=True)
    pairs = ground_truth_pairs()
    pairs = pairs[pairs["match_id"].isin(set(sample["entity_id"]))]
    s1 = load_table("train", "source1").select(["entity_id", "business_name"])
    s1 = s1.filter(pc.is_in(s1["entity_id"], value_set=pa.array(pairs["s1_id"].unique().tolist()))).to_pandas()
    df = (sample.merge(pairs, left_on="entity_id", right_on="match_id")
          .merge(s1.rename(columns={"entity_id": "s1_id", "business_name": "s1_name"}), on="s1_id"))
    target = [clean(x) for x in df["s1_name"]]
    print(f"{len(df):,} true pairs with an Indic-script name\n")
    rows = []
    for method, fn in METHODS.items():
        got = [clean(fn(x)) for x in df["business_name"]]
        tsr = process.cpdist(got, target, scorer=fuzz.token_sort_ratio, workers=-1)
        ratio = process.cpdist(got, target, scorer=fuzz.ratio, workers=-1)
        res = pd.DataFrame({"script": df["script"], "tsr": tsr, "ratio": ratio})
        rows.append({"method": method, "script": "ALL", "token_sort": tsr.mean(), "ratio": ratio.mean(),
                     "share_tsr_ge_80": (tsr >= 80).mean()})
        for sc, g in res.groupby("script"):
            rows.append({"method": method, "script": sc, "token_sort": g["tsr"].mean(), "ratio": g["ratio"].mean(),
                         "share_tsr_ge_80": (g["tsr"] >= 80).mean()})
    out = pd.DataFrame(rows)
    print(out[out["script"] == "ALL"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nmean token_sort_ratio by script:")
    print(out.pivot(index="script", columns="method", values="token_sort").round(1).to_string())
    ex = df.sample(n=8, random_state=SEED)
    print("\nexamples:")
    for r in ex.itertuples():
        print(f"  S1: {r.s1_name[:40]:<40} | ours: {clean(transliterate(r.business_name))[:40]:<40} | "
              f"iast: {clean(METHODS['sanscript_iast'](r.business_name))[:40]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
