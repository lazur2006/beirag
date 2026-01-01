"""
Wrapper for BEIR and sklearn integration BM25s framework.

Requires:
    pip install bm25s

Note:
    - Builds a Lucene index from a BEIR-style corpus if needed.
    - Uses BM25 with RM3 pseudo-relevance feedback:
        searcher.set_bm25(k1, b)
        searcher.set_rm3(fb_terms, fb_docs, original_query_weight)
"""

from typing import Dict, Any

from sklearn.base import BaseEstimator

from utils.custom_evaluation import EvaluateRetrievalCI

import bm25s


# ---------------------------------------------------------------------------
# BEIR wrapper
# ---------------------------------------------------------------------------

class BM25sSearch:
    def __init__(self, corpus: Dict[str, Dict[str, Any]], k1: float = 1.5, b: float = 0.75, delta:float = 0.5, method: str = "lucene"):
        # Stable mapping index -> doc_id
        self.doc_ids = list(corpus.keys())

        # Text that BM25 sees (title + text)
        self.bm25_corpus = [
            (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
            for doc_id in self.doc_ids
        ]

        # Tokenize and index with *no* corpus passed to BM25,
        # so retrieve() returns integer indices, not documents.
        corpus_tokens = bm25s.tokenize(self.bm25_corpus)
        self.retriever = bm25s.BM25(k1=k1, b=b, delta=delta, method=method)
        self.retriever.index(corpus_tokens)

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
        results: Dict[str, Dict[str, float]] = {}

        for qid, qtext in queries.items():
            # Tokenize single query
            query_tokens = bm25s.tokenize(qtext)

            # bm25s returns a NamedTuple (documents, scores)
            docs, scores = self.retriever.retrieve(query_tokens, k=top_k)
            # Since we didn't set self.corpus, docs == indices (2D array)
            doc_indices = docs[0]
            doc_scores = scores[0]

            results[qid] = {
                self.doc_ids[int(idx)]: float(score)
                for idx, score in zip(doc_indices, doc_scores)
            }

        return results
    

# ---------------------------------------------------------------------------
# Sklearn wrapper
# ---------------------------------------------------------------------------

class BM25sEstimator(BaseEstimator):
    def __init__(self, k1=1.5, b=0.75, delta=0.5, method="bm25+", k_eval=10):  # evaluate via NDCG@k_eval
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.method = method
        self.k_eval = k_eval

    def fit(self, X, y=None):
        # X is a list of {"corpus", "queries", "qrels"} dicts
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

        # build and index BM25sSearch
        self.model_ = BM25sSearch(
            self.corpus_,
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            method=self.method,
        )
        self.retriever_ = EvaluateRetrievalCI(self.model_)

        return self

    def score(self, X=None, y=None):
        results = self.retriever_.retrieve(self.corpus_, self.queries_)

        ndcg, _map, recall, precision, mrr = EvaluateRetrievalCI.evaluate(
            self.qrels_,
            results,
            k_values=[self.k_eval],
        )
        key = f"NDCG@{self.k_eval}"
        return ndcg[key]["mean"]
