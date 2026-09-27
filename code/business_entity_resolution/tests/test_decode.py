import pytest

from src.decode import best_k


@pytest.mark.parametrize(
    "ps,k",
    [
        ([0.6], 1),
        ([0.45, 0.30], 1),
        ([0.90, 0.50], 1),
        ([0.40, 0.40, 0.40], 3),
        ([0.55, 0.20, 0.20], 1),
        ([0.30], 0),
    ],
)
def test_matches_brute_force_table(ps, k):
    assert best_k(ps) == k
