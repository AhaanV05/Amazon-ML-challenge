"""Official metric: F0.5 per Source 1 entity, macro-averaged over all entities, singletons included."""
from collections.abc import Iterable, Mapping


def entity_f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    return 1.25 * tp / (0.25 * len(true) + len(pred))


def macro_f05(pred: Mapping[str, set], truth: Mapping[str, set], ids: Iterable[str] | None = None) -> float:
    keys = list(truth if ids is None else ids)
    return sum(entity_f05(set(pred.get(k, ())), truth[k]) for k in keys) / len(keys)


def oracle_ceiling(candidates: Mapping[str, set], truth: Mapping[str, set], ids: Iterable[str] | None = None) -> float:
    """Best macro F0.5 a perfect classifier could reach when restricted to these candidates."""
    keys = list(truth if ids is None else ids)
    total = 0.0
    for k in keys:
        t = truth[k]
        if not t:
            total += 1.0
            continue
        c = len(set(candidates.get(k, ())) & t)
        total += 1.25 * c / (c + 0.25 * len(t)) if c else 0.0
    return total / len(keys)


def scorecard(pred: Mapping[str, set], truth: Mapping[str, set], ids: Iterable[str] | None = None) -> dict:
    keys = list(truth if ids is None else ids)
    singles = [k for k in keys if not truth[k]]
    matched = [k for k in keys if truth[k]]
    return {
        "macro_f05": macro_f05(pred, truth, keys),
        "non_singleton_f05": macro_f05(pred, truth, matched) if matched else float("nan"),
        "singleton_accuracy": (sum(not pred.get(k) for k in singles) / len(singles)) if singles else float("nan"),
        "singleton_share": len(singles) / len(keys),
        "all_empty_baseline": len(singles) / len(keys),
    }
