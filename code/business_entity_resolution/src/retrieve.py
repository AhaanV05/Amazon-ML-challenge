"""Candidate retrieval: sparse TF-IDF top-k per country and source.

Every route is an inverted-index style sparse product restricted to one country, so cost grows
with the postings actually touched, never with |S1| x |S2|. Indexes are built once per country and
source; queries are searched in chunks so memory stays flat regardless of how many there are.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import THREADS

N_THREADS = THREADS
MAX_DF = 0.01  # tokens in >1% of records carry little identity and dominate cost
THRESHOLD = 0.1


def _name_place(phon: str, addr: str) -> str:
    """Composite keys: each phonetic name word paired with each alphabetic address word ("snrs@mumbai").

    City words are too common to search on alone, and common names are too; the pair is rare.
    """
    places = {t for t in addr.split() if t.isalpha() and len(t) >= 3}
    return " ".join(f"{p}@{a}" for p in set(phon.split()) for a in places)


# Name-only and phonetic-only routes were dropped: each found <0.05% of true pairs that no other route found.
ROUTES = {
    "place": lambda df: pd.Series([_name_place(p, a) for p, a in zip(df["name_phon"], df["addr_norm"])], index=df.index),
    "addr": lambda df: df["addr_norm"],
    "joint": lambda df: ("n_" + df["name_core"].str.replace(" ", " n_", regex=False)).where(df["name_core"] != "", "")
    + " " + df["addr_norm"],
}
HARD = "hard"
HARD_K = 10
ROUTE_NAMES = (*ROUTES, HARD)


def _topk(a: sparse.csr_matrix, bt: sparse.csr_matrix, k: int, threshold: float) -> tuple[np.ndarray, ...]:
    m = sp_matmul_topn(a, bt, top_n=k, threshold=threshold, sort=True, n_threads=N_THREADS)
    rows = np.repeat(np.arange(m.shape[0], dtype=np.int64), np.diff(m.indptr))
    rank = (np.arange(m.nnz, dtype=np.int64) - m.indptr[rows]).astype(np.float32)
    return rows, m.indices.astype(np.int64), m.data.astype(np.float32), rank


class RouteIndex:
    """One route's TF-IDF index over the searched records (weights fitted on those records only)."""

    def __init__(self, route: str, docs: pd.DataFrame, max_df: float = MAX_DF):
        self.route = route
        self.vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, dtype=np.float32,
                                   max_df=max_df, min_df=1)
        self.dm = self.vec.fit_transform(ROUTES[route](docs))
        self.dt = self.dm.T.tocsr()

    def query_matrix(self, queries: pd.DataFrame) -> sparse.csr_matrix:
        return self.vec.transform(ROUTES[self.route](queries))

    def search(self, queries: pd.DataFrame, k: int, threshold: float = THRESHOLD) -> pd.DataFrame:
        q, d, score, rank = _topk(self.query_matrix(queries), self.dt, k, threshold)
        return pd.DataFrame({"q": q, "d": d, f"{self.route}_score": score, f"{self.route}_rank": rank})


def _squash(names: pd.Series) -> pd.Series:
    return names.str.replace(" ", "", regex=False)


def _hard_docs(docs: pd.DataFrame) -> np.ndarray:
    return ((docs["addr_norm"] == "") | docs["name_domain"] | docs["name_indic"]).to_numpy()


class HardNameIndex:
    """Name-only char-3-gram search over the docs the word routes miss most: no address, a domain-style name
    ("dynamicsystems.com"), or an Indic-script name. Indexing only those docs keeps a name alone from being
    crowded out; on 40k train entities this lifted the retrieval ceiling from 0.9928 to 0.9953 at top-10."""

    route = HARD

    def __init__(self, docs: pd.DataFrame):
        self.rows = np.flatnonzero(_hard_docs(docs))
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32,
                                   max_df=0.02)
        self.dt = self.vec.fit_transform(_squash(docs["name_core"].iloc[self.rows])).T.tocsr()

    def search(self, queries: pd.DataFrame, k: int, threshold: float = THRESHOLD) -> pd.DataFrame:
        q, d, score, rank = _topk(self.vec.transform(_squash(queries["name_core"])), self.dt, min(k, HARD_K),
                                  threshold)
        return pd.DataFrame({"q": q, "d": self.rows[d], f"{HARD}_score": score, f"{HARD}_rank": rank})


def build_indexes(docs: pd.DataFrame, routes=ROUTE_NAMES, max_df: float = MAX_DF) -> list:
    out = [RouteIndex(r, docs, max_df) for r in routes if r != HARD]
    if HARD in routes and _hard_docs(docs).any():
        out.append(HardNameIndex(docs))
    return out


def search_all(indexes: list[RouteIndex], queries: pd.DataFrame, k: int, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Union of every route's top-k for these queries; q indexes rows of `queries`."""
    out = None
    for idx in indexes:
        part = idx.search(queries, k, threshold)
        out = part if out is None else out.merge(part, on=["q", "d"], how="outer")
    return out


def retrieve(queries: pd.DataFrame, docs: pd.DataFrame, k: int, k_rev: int = 0, threshold: float = THRESHOLD,
             max_df: float = MAX_DF, routes: tuple[str, ...] = tuple(ROUTES)) -> pd.DataFrame:
    """All-at-once retrieval (evaluation helper), optionally with reverse search from docs to queries."""
    indexes = build_indexes(docs, routes, max_df)
    out = search_all(indexes, queries, k, threshold)
    if k_rev:
        for idx in indexes:
            d, q, score, rank = _topk(idx.dm, idx.query_matrix(queries).T.tocsr(), k_rev, threshold)
            rev = pd.DataFrame({"q": q, "d": d, f"{idx.route}_rscore": score, f"{idx.route}_rrank": rank})
            out = out.merge(rev, on=["q", "d"], how="outer")
    return out
