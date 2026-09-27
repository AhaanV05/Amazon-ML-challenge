"""Write matching_results.tsv and candidate_pairs.tsv exactly as the organizers specify."""
from collections.abc import Iterable, Mapping
from pathlib import Path


def _sort_key(entity_id: str) -> tuple[str, int]:
    return entity_id[:3], int(entity_id[3:])


def write_id_lists(path: Path, s1_ids: Iterable[str], lists: Mapping[str, Iterable[str]], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{column}\n")
        for s1 in s1_ids:
            ids = sorted(set(lists.get(s1, ())), key=_sort_key)
            f.write(f"{s1}\t{','.join(ids)}\n")


def write_submission(out_dir: Path, s1_ids: list[str], matches: Mapping[str, Iterable[str]],
                     candidates: Mapping[str, Iterable[str]]) -> None:
    for s1, m in matches.items():
        missing = set(m) - set(candidates.get(s1, ()))
        if missing:
            raise ValueError(f"{s1}: matched ids not in candidates: {sorted(missing)[:5]}")
    write_id_lists(out_dir / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    write_id_lists(out_dir / "candidate_pairs.tsv", s1_ids, candidates, "candidate_entity_ids")
