"""
Wrapper for BEIR and sklearn integration of Pyserini BM25+RM3.

Usage pattern is analogous to `bm25_wrapper.py`:

- `PyseriniBM25RM3Search`  -> BEIR-style `.search(...)` method
- `PyseriniBM25RM3Estimator` -> sklearn-style estimator using EvaluateRetrievalCI

Indexing is done inside the wrapper.

If Java / Pyserini are not properly installed, importing this module will
NOT crash; instead, instantiating the classes will raise a clear RuntimeError.
"""

import os
from typing import Dict, Any

from sklearn.base import BaseEstimator
from utils.custom_evaluation import EvaluateRetrievalCI

# ---------------------------------------------------------------------------
# Safe import of Pyserini (and Java)
# ---------------------------------------------------------------------------

_PYSERINI_AVAILABLE = False
_PYSERINI_IMPORT_ERROR = None

try:
    from pyserini.index.lucene import LuceneIndexer
    from pyserini.search.lucene import LuceneSearcher
    _PYSERINI_AVAILABLE = True
except Exception as e:  # catches jnius Java errors as well
    LuceneIndexer = None  # type: ignore
    LuceneSearcher = None  # type: ignore
    _PYSERINI_AVAILABLE = False
    _PYSERINI_IMPORT_ERROR = e


def _ensure_pyserini_available():
    if not _PYSERINI_AVAILABLE:
        msg = [
            "PyseriniBM25RM3 requires a working Java + Pyserini installation.",
            "",
            "Original import error from Pyserini/jnius:",
            f"  {repr(_PYSERINI_IMPORT_ERROR)}",
            "",
            "Fix suggestions:",
            "  1) Install a JDK (e.g. `sudo apt-get install openjdk-11-jdk`)",
            "  2) Set JAVA_HOME, e.g.:",
            "       export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64",
            "       export PATH=\"$JAVA_HOME/bin:$PATH\"",
            "",
            "  3) Ensure `pyserini` is installed in this environment:",
            "       pip install pyserini",
        ]
        raise RuntimeError("\n".join(msg))


# ---------------------------------------------------------------------------
# BEIR wrapper
# ---------------------------------------------------------------------------

class PyseriniBM25RM3Search:
    def __init__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        index_dir: str = "lucene-index",
        k1: float = 0.9,
        b: float = 0.4,
        fb_terms: int = 10,
        fb_docs: int = 10,
        original_query_weight: float = 0.5,
        rebuild_index: bool = False,
        index_threads: int = 8,  # kept for API symmetry; not used directly
    ):
        """
        BEIR-compatible searcher based on Pyserini BM25 + RM3.

        Args:
            corpus: BEIR-style corpus dict[doc_id] -> {"title": ..., "text": ...}
            index_dir: Directory where the Lucene index is stored.
            k1, b: BM25 parameters.
            fb_terms, fb_docs, original_query_weight:
                RM3 pseudo-relevance feedback parameters.
            rebuild_index: If True, (re)build the index even if it exists.
            index_threads: Present for symmetry; LuceneIndexer itself is single-threaded.
        """
        _ensure_pyserini_available()

        self.index_dir = index_dir

        # Build index if needed
        if rebuild_index or not os.path.exists(index_dir) or not os.listdir(index_dir):
            self._build_index(corpus, index_dir=index_dir)

        # Create searcher and configure BM25 + RM3
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(k1=k1, b=b)
        self.searcher.set_rm3(
            fb_terms=fb_terms,
            fb_docs=fb_docs,
            original_query_weight=original_query_weight,
        )

    @staticmethod
    def _build_index(
        corpus: Dict[str, Dict[str, Any]],
        index_dir: str,
    ) -> None:
        """
        Build a Lucene index from a BEIR corpus.

        Each BEIR doc:
            id       = BEIR doc_id
            contents = title + " " + text

        Index is created with stored contents & docvectors so RM3 can work.
        """
        _ensure_pyserini_available()

        os.makedirs(index_dir, exist_ok=True)

        # These args correspond to Anserini's IndexCollection flags.
        args = ["-index", index_dir, "-storeContents", "-storeDocvectors"]
        indexer = LuceneIndexer(index_dir=index_dir, args=args)

        n_docs = 0
        for doc_id, fields in corpus.items():
            contents = (
                fields.get("title", "") + " " + fields.get("text", "")
            ).strip()
            if not contents:
                continue
            indexer.add_doc_dict({"id": doc_id, "contents": contents})
            n_docs += 1

        indexer.close()
        print(
            f"[PyseriniBM25RM3Search] Built Lucene index with {n_docs} documents "
            f"at '{index_dir}'."
        )

    def search(
        self,
        corpus: Dict[str, Dict[str, Any]],   # BEIR passes this, we don't need it
        queries: Dict[str, str],
        top_k: int,
        score_function: str = None,         # ignored for lexical
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        BEIR-style search interface.

        Returns:
            results[qid][doc_id] = score
        """
        _ensure_pyserini_available()

        results: Dict[str, Dict[str, float]] = {}

        # Per-query loop, analogous to BM25sSearch
        for qid, qtext in queries.items():
            hits = self.searcher.search(qtext, k=top_k)
            results[qid] = {h.docid: float(h.score) for h in hits}

        return results


# ---------------------------------------------------------------------------
# Sklearn wrapper
# ---------------------------------------------------------------------------

class PyseriniBM25RM3Estimator(BaseEstimator):
    """
    Sklearn-compatible estimator for Pyserini BM25+RM3.

    Mirrors BM25sEstimator:

      - fit()   -> builds/loads Lucene index, wraps with EvaluateRetrievalCI
      - score() -> returns mean NDCG@k_eval over dev set(s)
    """

    def __init__(
        self,
        index_dir: str = "lucene-index",
        k1: float = 0.9,
        b: float = 0.4,
        fb_terms: int = 10,
        fb_docs: int = 10,
        original_query_weight: float = 0.5,
        k_eval: int = 10,
        rebuild_index: bool = False,
        index_threads: int = 8,
    ):
        """
        Args:
            index_dir: Directory where the Lucene index is stored/built.
            k1, b: BM25 parameters.
            fb_terms, fb_docs, original_query_weight: RM3 parameters.
            k_eval: cutoff for NDCG@k_eval used as sklearn scoring metric.
            rebuild_index: If True, always rebuild the index in fit().
            index_threads: Kept for API symmetry; not used directly.
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.fb_terms = fb_terms
        self.fb_docs = fb_docs
        self.original_query_weight = original_query_weight
        self.k_eval = k_eval
        self.rebuild_index = rebuild_index
        self.index_threads = index_threads

    def fit(self, X, y=None):
        """
        X can be:
            - a single dict: {"corpus", "queries", "qrels"}
            - or a list of such dicts (will be merged).
        """
        _ensure_pyserini_available()

        if isinstance(X, dict):
            datasets = [X]
        else:
            datasets = X

        corpus = {}
        queries = {}
        qrels = {}
        for d in datasets:
            corpus.update(d["corpus"])
            queries.update(d["queries"])
            qrels.update(d["qrels"])

        self.corpus_ = corpus
        self.queries_ = queries
        self.qrels_ = qrels

        # Build / load BM25+RM3 searcher on Lucene index
        self.model_ = PyseriniBM25RM3Search(
            self.corpus_,
            index_dir=self.index_dir,
            k1=self.k1,
            b=self.b,
            fb_terms=self.fb_terms,
            fb_docs=self.fb_docs,
            original_query_weight=self.original_query_weight,
            rebuild_index=self.rebuild_index,
            index_threads=self.index_threads,
        )
        self.retriever_ = EvaluateRetrievalCI(self.model_)

        return self

    def score(self, X=None, y=None):
        """
        Runs retrieval on self.corpus_/self.queries_ and returns mean NDCG@k_eval.
        """
        _ensure_pyserini_available()

        results = self.retriever_.retrieve(self.corpus_, self.queries_)

        ndcg, _map, recall, precision, mrr = EvaluateRetrievalCI.evaluate(
            self.qrels_,
            results,
            k_values=[self.k_eval],
        )
        key = f"NDCG@{self.k_eval}"
        return ndcg[key]["mean"]
