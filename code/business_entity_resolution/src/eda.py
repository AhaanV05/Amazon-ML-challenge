"""Dataset exploration. Run: python -m src.eda  (from code/business_entity_resolution)."""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.metrics import roc_auc_score

from .config import SEED, WORK_DIR
from .data import load_table
from .text import basic_norm, numbers, postcode

OUT = WORK_DIR / "eda"
REPORT: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    REPORT.append(line)


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


PATTERNS = {
    "name_devanagari": ("business_name", r"\p{Devanagari}"),
    "addr_devanagari": ("business_address", r"\p{Devanagari}"),
    "name_leading_junk": ("business_name", r"^[^\p{L}\p{N}]+"),
    "name_domain": ("business_name", r"(?i)\.(com|net|org|in|biz|info|fr|io|us)$"),
    "name_legal_first": ("business_name", r"(?i)^\W*(llc|inc|ltd|pvt|corp|co|sarl|sas|sa|sci|eurl|llp|plc)\b"),
    "addr_landmark": ("business_address", r"(?i)\b(near|nr|opp|opposite|behind|beside|next to)\b"),
    "addr_5digit": ("business_address", r"(^|\D)\d{5}(\D|$)"),
    "addr_6digit": ("business_address", r"(^|\D)(\d{6}|\d{3} \d{3})(\D|$)"),
}


def file_profile(split: str, name: str) -> dict:
    t = load_table(split, name)
    n = t.num_rows
    prof = {"rows": n}
    prof["country"] = {r["values"]: r["counts"] for r in pc.value_counts(t["country"]).to_pylist()}
    for col in ("business_name", "business_address"):
        lens = pc.utf8_length(t[col]).to_numpy()
        prof[f"{col}_empty"] = float((lens == 0).mean())
        prof[f"{col}_len_p50_p95"] = [int(np.percentile(lens, 50)), int(np.percentile(lens, 95))]
    for key, (col, rx) in PATTERNS.items():
        prof[key] = float(pc.mean(pc.cast(pc.match_substring_regex(t[col], rx), pa.int8())).as_py())
    addr = t["business_address"]
    upper = pc.and_(pc.equal(addr, pc.utf8_upper(addr)), pc.match_substring_regex(addr, "[A-Za-z]"))
    prof["addr_all_upper"] = float(pc.mean(pc.cast(upper, pa.int8())).as_py())
    key = pc.binary_join_element_wise(pc.utf8_lower(t["business_name"]), pc.utf8_lower(addr), "|")
    prof["exact_dup_rows"] = 1 - pc.count_distinct(key).as_py() / n
    return prof


def section_files() -> dict:
    say("=" * 100)
    say("1. PER-FILE PROFILE")
    profiles = {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            t0 = time.time()
            p = file_profile(split, f"source{s}")
            profiles[f"{split}_s{s}"] = p
            say(f"\n{split} source{s}: {p['rows']:,} rows  ({time.time() - t0:.0f}s)")
            say("  country: " + ", ".join(f"{k}={v:,} ({pct(v / p['rows'])})" for k, v in p["country"].items()))
            say(f"  empty name {pct(p['business_name_empty'])} | empty address {pct(p['business_address_empty'])}"
                f" | name len p50/p95 {p['business_name_len_p50_p95']} | addr len p50/p95 {p['business_address_len_p50_p95']}")
            say("  " + " | ".join(f"{k} {pct(p[k])}" for k in PATTERNS) + f" | addr_all_upper {pct(p['addr_all_upper'])}")
            say(f"  exact duplicate (name,address) rows {pct(p['exact_dup_rows'])}")
    return profiles


def section_ground_truth() -> dict:
    say("\n" + "=" * 100)
    say("2. GROUND TRUTH STRUCTURE")
    gt = load_table("train", "ground_truth")
    s1_ids = load_table("train", "source1")["entity_id"].combine_chunks()
    n = gt.num_rows
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    flat = pc.list_flatten(lists)
    parents = pc.list_parent_indices(lists).to_numpy()
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    flat, parents = flat.combine_chunks().filter(pa.array(keep)), parents[keep]
    prefix = pc.utf8_slice_codeunits(flat, 0, 3).to_numpy(zero_copy_only=False)
    n_all = np.bincount(parents, minlength=n)
    n_s2 = np.bincount(parents[prefix == "S2-"], minlength=n)
    n_s3 = np.bincount(parents[prefix == "S3-"], minlength=n)
    res = {"gt_rows": n, "s1_rows": len(s1_ids)}
    res["gt_ids_equal_s1_ids"] = bool(pc.all(pc.is_in(gt["source1_entity_id"], value_set=s1_ids)).as_py()) and n == len(s1_ids)
    res["singleton_rate"] = float((n_all == 0).mean())
    say(f"rows {n:,} | same id set as source1: {res['gt_ids_equal_s1_ids']} | singletons {pct(res['singleton_rate'])}")
    say(f"pairs {len(flat):,} | mean matches per entity {n_all.mean():.2f} (non-singletons {n_all[n_all > 0].mean():.2f})")

    def dist(arr, label):
        bins = [0, 1, 2, 3, 4, 5, 6, 11, 10**9]
        names = ["0", "1", "2", "3", "4", "5", "6-10", ">10"]
        counts = np.histogram(arr, bins=bins)[0] / len(arr)
        say(f"  {label:<14}" + "  ".join(f"{nm}:{pct(c)}" for nm, c in zip(names, counts)) + f"  max {arr.max()}")
        return dict(zip(names, counts.tolist()))

    res["dist_all"] = dist(n_all, "all matches")
    res["dist_s2"] = dist(n_s2, "S2 matches")
    res["dist_s3"] = dist(n_s3, "S3 matches")
    both = ((n_s2 > 0) & (n_s3 > 0)).mean()
    only2 = ((n_s2 > 0) & (n_s3 == 0)).mean()
    only3 = ((n_s2 == 0) & (n_s3 > 0)).mean()
    say(f"  entity has S2 and S3 {pct(both)} | only S2 {pct(only2)} | only S3 {pct(only3)}")

    vc = pc.value_counts(flat)
    multi = pc.sum(pc.cast(pc.greater(vc.field("counts"), 1), pa.int64())).as_py()
    res["ids_under_multiple_s1"] = multi
    say(f"  matched ids appearing under more than one S1 entity: {multi:,}")

    for s in (2, 3):
        ids = load_table("train", f"source{s}")["entity_id"]
        covered = pc.mean(pc.cast(pc.is_in(ids, value_set=flat), pa.int8())).as_py()
        res[f"s{s}_records_in_gt"] = covered
        say(f"  share of train S{s} records that belong to some S1 entity: {pct(covered)} (rest are distractors)")
    all_ids = pa.concat_arrays([load_table("train", f"source{s}")["entity_id"].combine_chunks() for s in (2, 3)])
    res["gt_ids_missing_from_sources"] = int(len(flat) - pc.sum(pc.cast(pc.is_in(flat, value_set=all_ids), pa.int64())).as_py())
    say(f"  matched ids missing from source files: {res['gt_ids_missing_from_sources']:,}")
    return res


def _lookup(split: str, source: int, ids: set[str]) -> pd.DataFrame:
    t = load_table(split, f"source{source}")
    t = t.filter(pc.is_in(t["entity_id"], value_set=pa.array(sorted(ids))))
    return t.to_pandas().set_index("entity_id")


def sample_pairs(n_entities: int, rng: np.random.Generator):
    gt = load_table("train", "ground_truth").to_pandas()
    has = gt[gt["matched_entity_ids"] != ""]
    sample = has.sample(n=n_entities, random_state=SEED)
    pairs = sample.assign(match_id=sample["matched_entity_ids"].str.split(",")).explode("match_id")
    pairs = pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "match_id"]]
    singles = gt[gt["matched_entity_ids"] == ""]["source1_entity_id"].sample(n=min(2000, (gt["matched_entity_ids"] == "").sum()), random_state=SEED)
    return pairs, set(sample["source1_entity_id"]), set(singles), gt


def section_pairs(rng: np.random.Generator) -> dict:
    say("\n" + "=" * 100)
    say("3. PAIR-LEVEL AGREEMENT ON TRUE MATCHES (sample)")
    pairs, s1_sample, singles, gt = sample_pairs(30000, rng)
    s1 = _lookup("train", 1, s1_sample | singles)
    recs = pd.concat([_lookup("train", s, set(pairs["match_id"])) for s in (2, 3)])
    p = pairs.join(s1.add_prefix("a_"), on="s1_id").join(recs.add_prefix("b_"), on="match_id")
    p["src"] = p["match_id"].str[:2]
    res = {"n_pairs": len(p)}
    res["country_agree"] = float((p["a_country"] == p["b_country"]).mean())
    zc_a = p["a_business_address"].map(postcode)
    zc_b = p["b_business_address"].map(postcode)
    both = (zc_a != "") & (zc_b != "")
    res["postcode_both_present"] = float(both.mean())
    res["postcode_equal_when_both"] = float((zc_a[both] == zc_b[both]).mean())
    na, nb = p["a_business_address"].map(numbers), p["b_business_address"].map(numbers)
    has_nums = (na.map(len) > 0) & (nb.map(len) > 0)
    res["numbers_overlap_when_both"] = float(pd.Series([len(x & y) > 0 for x, y in zip(na[has_nums], nb[has_nums])]).mean())
    res["name_exact_norm"] = float((p["a_business_name"].map(basic_norm) == p["b_business_name"].map(basic_norm)).mean())
    res["addr_exact_norm"] = float((p["a_business_address"].map(basic_norm) == p["b_business_address"].map(basic_norm)).mean())
    dev = p["b_business_name"].str.contains(r"[ऀ-ॿ]", regex=True)
    res["match_name_devanagari"] = float(dev.mean())
    res["s1_name_devanagari"] = float(p["a_business_name"].str.contains(r"[ऀ-ॿ]", regex=True).mean())
    for k, v in res.items():
        say(f"  {k:<30} {v if isinstance(v, int) else pct(v)}")
    for src in ("S2", "S3"):
        q = p[p["src"] == src]
        say(f"  {src}: country agree {pct((q['a_country'] == q['b_country']).mean())} | "
            f"name exact(norm) {pct((q['a_business_name'].map(basic_norm) == q['b_business_name'].map(basic_norm)).mean())} | "
            f"addr empty {pct((q['b_business_address'] == '').mean())}")
    for c in ("US", "India"):
        q = p[p["a_country"] == c]
        z_a, z_b = q["a_business_address"].map(postcode), q["b_business_address"].map(postcode)
        say(f"  {c}: postcode present S1 {pct((z_a != '').mean())} / match {pct((z_b != '').mean())} | "
            f"equal when both {pct((z_a[(z_a != '') & (z_b != '')] == z_b[(z_a != '') & (z_b != '')]).mean())}")
    return res, p, s1, singles


def _features(a_name, b_name, a_addr, b_addr) -> pd.DataFrame:
    an, bn = [basic_norm(x) for x in a_name], [basic_norm(x) for x in b_name]
    aa, ba = [basic_norm(x) for x in a_addr], [basic_norm(x) for x in b_addr]
    f = pd.DataFrame({
        "name_ratio": process.cpdist(an, bn, scorer=fuzz.ratio, workers=-1),
        "name_token_sort": process.cpdist(an, bn, scorer=fuzz.token_sort_ratio, workers=-1),
        "name_token_set": process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1),
        "name_partial": process.cpdist(an, bn, scorer=fuzz.partial_ratio, workers=-1),
        "name_jaro_winkler": process.cpdist(an, bn, scorer=JaroWinkler.normalized_similarity, workers=-1),
        "addr_ratio": process.cpdist(aa, ba, scorer=fuzz.ratio, workers=-1),
        "addr_token_set": process.cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=-1),
    })
    za, zb = [postcode(x) for x in a_addr], [postcode(x) for x in b_addr]
    f["postcode_state"] = [0 if not x or not y else (1 if x == y else -1) for x, y in zip(za, zb)]
    nums = [(numbers(x), numbers(y)) for x, y in zip(a_addr, b_addr)]
    f["number_jaccard"] = [len(x & y) / len(x | y) if x and y else 0.5 for x, y in nums]
    f["first_token_equal"] = [int(bool(x) and bool(y) and x.split()[0] == y.split()[0]) for x, y in zip(an, bn)]
    return f


def section_separability(p: pd.DataFrame, rng: np.random.Generator) -> dict:
    say("\n" + "=" * 100)
    say("4. FEATURE SEPARABILITY: TRUE PAIRS vs NEGATIVES (univariate AUC)")
    pos = p.groupby("s1_id").head(3).reset_index(drop=True)
    pos = pos.sample(n=min(40000, len(pos)), random_state=SEED).reset_index(drop=True)
    true_sets = p.groupby("s1_id")["match_id"].agg(set)

    negs = []
    for s in (2, 3):
        t = load_table("train", f"source{s}")
        idx = rng.choice(t.num_rows, size=600_000, replace=False)
        pool = t.take(pa.array(idx)).to_pandas()
        pool["first"] = pool["business_name"].map(lambda x: (basic_norm(x).split() or [""])[0])
        pool["state"] = pool["business_address"].map(lambda x: basic_norm(x.split(",")[-1]) if x else "")
        negs.append(pool)
    pool = pd.concat(negs, ignore_index=True)
    by_first = pool.groupby(["country", "first"]).indices
    by_country = pool.groupby("country").indices

    rows_hard, rows_rand = [], []
    for r in pos.itertuples():
        first = (basic_norm(r.a_business_name).split() or [""])[0]
        cand = by_first.get((r.a_country, first))
        if cand is not None and len(cand) > 1:
            j = cand[rng.integers(len(cand))]
            if pool.at[j, "entity_id"] not in true_sets[r.s1_id]:
                rows_hard.append((r.Index, j))
        cc = by_country[r.a_country]
        j = cc[rng.integers(len(cc))]
        if pool.at[j, "entity_id"] not in true_sets[r.s1_id]:
            rows_rand.append((r.Index, j))

    def build(rows, label):
        a = pos.loc[[i for i, _ in rows], ["a_business_name", "a_business_address", "a_country"]].reset_index(drop=True)
        b = pool.loc[[j for _, j in rows], ["business_name", "business_address"]].reset_index(drop=True)
        return pd.concat([a, b.add_prefix("b_")], axis=1).assign(kind=label)

    hard, rand = build(rows_hard, "hard_neg"), build(rows_rand, "random_neg")
    allp = pd.concat([pos[["a_business_name", "a_business_address", "a_country", "b_business_name", "b_business_address"]].assign(kind="pos"),
                      hard, rand], ignore_index=True)
    feats = _features(allp["a_business_name"], allp["b_business_name"], allp["a_business_address"], allp["b_business_address"])
    feats["kind"], feats["country"] = allp["kind"].values, allp["a_country"].values
    say(f"pairs: pos {len(pos):,} | hard neg (same country + same first name token) {len(hard):,} | random neg (same country) {len(rand):,}")
    say(f"  {'feature':<20}{'AUC vs hard':>12}{'AUC vs rand':>12}{'pos mean':>10}{'hard mean':>10}{'rand mean':>10}"
        f"{'AUC hard US':>13}{'AUC hard IN':>13}")
    res = {}
    cols = [c for c in feats.columns if c not in ("kind", "country")]
    for c in cols:
        def auc(neg, country=None):
            m = feats["kind"].isin(["pos", neg])
            if country:
                m &= feats["country"] == country
            y = (feats.loc[m, "kind"] == "pos").astype(int)
            return roc_auc_score(y, feats.loc[m, c]) if y.nunique() == 2 else float("nan")
        means = feats.groupby("kind")[c].mean()
        res[c] = {"auc_hard": auc("hard_neg"), "auc_rand": auc("random_neg")}
        say(f"  {c:<20}{res[c]['auc_hard']:>12.3f}{res[c]['auc_rand']:>12.3f}{means['pos']:>10.2f}{means['hard_neg']:>10.2f}"
            f"{means['random_neg']:>10.2f}{auc('hard_neg', 'US'):>13.3f}{auc('hard_neg', 'India'):>13.3f}")
    corr = feats.loc[feats["kind"].isin(["pos", "hard_neg"]), cols].corr(method="spearman")
    say("\n  Spearman correlation between features (pos + hard neg):")
    say("  " + " " * 18 + "".join(f"{c[:8]:>9}" for c in cols))
    for c in cols:
        say(f"  {c:<18}" + "".join(f"{corr.loc[c, d]:>9.2f}" for d in cols))
    feats.to_parquet(OUT / "separability_features.parquet")
    return res


def section_examples(p: pd.DataFrame, s1: pd.DataFrame, singles: set[str], rng: np.random.Generator) -> None:
    say("\n" + "=" * 100)
    say("5. EXAMPLES")

    def row(tag, name, addr, country):
        say(f"    {tag:<4} {name[:52]:<52} | {addr[:70]:<70} | {country}")

    for country in ("US", "India"):
        ids = p[p["a_country"] == country]["s1_id"].drop_duplicates().sample(n=4, random_state=SEED)
        for sid in ids:
            q = p[p["s1_id"] == sid]
            say(f"  {sid} ({len(q)} matches)")
            first = q.iloc[0]
            row("S1", first["a_business_name"], first["a_business_address"], first["a_country"])
            for r in q.itertuples():
                row(r.src, r.b_business_name, r.b_business_address, r.b_country)
    say("  singletons:")
    for sid in list(singles)[:4]:
        r = s1.loc[sid]
        row("S1", r["business_name"], r["business_address"], r["country"])
    say("  France in test:")
    for s in (1, 2, 3):
        t = load_table("test", f"source{s}")
        fr = t.filter(pc.equal(t["country"], "France"))
        take = fr.take(pa.array(rng.choice(fr.num_rows, size=3, replace=False))).to_pylist()
        for r in take:
            row(f"S{s}", r["business_name"], r["business_address"], r["country"])


def section_id_leakage(p: pd.DataFrame) -> dict:
    say("\n" + "=" * 100)
    say("6. ID LEAKAGE CHECK")
    a = p["s1_id"].str[3:].astype("int64")
    b = p["match_id"].str[3:].astype("int64")
    rho = pd.Series(a.values).corr(pd.Series(b.values), method="spearman")
    say(f"  Spearman(S1 id number, matched id number) on {len(p):,} true pairs: {rho:.4f} (near 0 means no leak)")
    return {"id_spearman": float(rho)}


SCRIPTS = {
    "Devanagari": "ऀ-ॿ", "Bengali": "ঀ-৿", "Gurmukhi": "਀-੿",
    "Gujarati": "઀-૿", "Oriya": "଀-୿", "Tamil": "஀-௿", "Telugu": "ఀ-౿",
    "Kannada": "ಀ-೿", "Malayalam": "ഀ-ൿ", "Arabic": "؀-ۿ", "CJK": "一-鿿",
}


def section_extra() -> None:
    say("\n" + "=" * 100)
    say("7. ADDRESS UNIQUENESS, SCRIPTS, UNUSABLE NAMES")
    for split in ("train", "test"):
        t = load_table(split, "source1")
        df = t.select(["business_address", "country"]).to_pandas()
        df["norm"] = df["business_address"].map(basic_norm)
        for c, g in df.groupby("country"):
            dup = 1 - g["norm"].nunique() / len(g)
            say(f"  {split} S1 {c:<7} share of entities whose normalized address is shared with another S1 entity: "
                f"{pct(g['norm'].duplicated(keep=False).mean())} (distinct-address deficit {pct(dup)})")
    for split in ("train", "test"):
        for s in (2, 3):
            t = load_table(split, f"source{s}")
            parts = []
            for col in ("business_name", "business_address"):
                shares = {k: pc.mean(pc.cast(pc.match_substring_regex(t[col], f"[{rng}]"), pa.int8())).as_py()
                          for k, rng in SCRIPTS.items()}
                parts.append(f"{col[9:]}: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in shares.items() if v > 0))
            say(f"  {split} S{s} scripts -> " + " || ".join(parts))
    feats = pd.read_parquet(OUT / "separability_features.parquet")
    pos = feats[feats["kind"] == "pos"]
    for c in ("US", "India"):
        q = pos[pos["country"] == c]
        say(f"  {c}: true pairs with name token-set < 40: {pct((q['name_token_set'] < 40).mean())} | "
            f"with address token-set < 40: {pct((q['addr_token_set'] < 40).mean())} | both < 40: "
            f"{pct(((q['name_token_set'] < 40) & (q['addr_token_set'] < 40)).mean())}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        section_extra()
        (OUT / "report_extra.txt").write_text("\n".join(REPORT), encoding="utf-8")
        return
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    results = {"files": section_files(), "ground_truth": section_ground_truth()}
    res_pairs, p, s1, singles = section_pairs(rng)
    results["pairs"] = res_pairs
    results["separability"] = section_separability(p, rng)
    section_examples(p, s1, singles, rng)
    results["id_leakage"] = section_id_leakage(p)
    say(f"\nTotal {time.time() - t0:.0f}s")
    (OUT / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
