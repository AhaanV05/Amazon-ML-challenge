import pytest

from src.submission import write_submission


def test_writes_exact_format(tmp_path):
    s1 = ["S1-2", "S1-10", "S1-3"]
    matches = {"S1-2": ["S3-5", "S2-40", "S2-7"], "S1-10": []}
    cands = {"S1-2": ["S2-7", "S2-40", "S3-5", "S3-99"], "S1-10": ["S2-1"]}
    write_submission(tmp_path, s1, matches, cands)
    m = (tmp_path / "matching_results.tsv").read_bytes().decode()
    c = (tmp_path / "candidate_pairs.tsv").read_bytes().decode()
    assert m == "source1_entity_id\tmatched_entity_ids\nS1-2\tS2-7,S2-40,S3-5\nS1-10\t\nS1-3\t\n"
    assert c == "source1_entity_id\tcandidate_entity_ids\nS1-2\tS2-7,S2-40,S3-5,S3-99\nS1-10\tS2-1\nS1-3\t\n"
    assert "\r" not in m


def test_rejects_match_outside_candidates(tmp_path):
    with pytest.raises(ValueError):
        write_submission(tmp_path, ["S1-1"], {"S1-1": ["S2-9"]}, {"S1-1": ["S2-1"]})
