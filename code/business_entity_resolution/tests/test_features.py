import numpy as np
import pandas as pd

from src.features import pair_features


def _frame(names, addrs):
    n = len(names)
    return pd.DataFrame({
        "name_core": names, "name_phon": [x[:3] for x in names], "addr_norm": addrs,
        "addr_nums": ["12 5", "7", "", "12"][:n], "name_legal": ["llc", "", "inc", "llc"][:n],
        "postcode": ["", "12345", "", "54321"][:n], "name_domain": [False] * n,
        "name_indic": [False] * n, "addr_indic": [False] * n,
    })


def test_chunking_does_not_change_features():
    q = _frame(["alpha beta", "gamma", "delta co", "eps"], ["12 main street", "7 oak road", "x", "12 elm"])
    d = _frame(["alpha", "gama", "delta", "zeta"], ["12 main st", "7 oak rd", "", "12 elm"])
    rng = np.random.default_rng(0)
    n = 23
    cand = pd.DataFrame({
        "q": rng.integers(0, 4, n), "d": rng.integers(0, 4, n), "source": np.full(n, 2, np.int8),
        "place_score": rng.random(n).astype(np.float32), "place_rank": rng.integers(0, 5, n).astype(np.float32),
        "addr_score": rng.random(n).astype(np.float32), "addr_rank": rng.integers(0, 5, n).astype(np.float32),
        "joint_score": rng.random(n).astype(np.float32), "joint_rank": rng.integers(0, 5, n).astype(np.float32),
    })
    whole = pair_features(cand, q, d, chunk=1000)
    chunked = pair_features(cand, q, d, chunk=4)
    pd.testing.assert_frame_equal(whole, chunked)
