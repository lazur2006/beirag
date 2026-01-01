"""
Reciprocal Rank Fusion for combining multiple precomputed retrieval runs.

Input shape for each run:
    results[qid][doc_id] = score

Fusion uses rank positions, so score scales can differ across runs. Gains
(weights) let you emphasize some runs more than others.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence

from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from utils.custom_evaluation import EvaluateRetrievalCI


ResultDict = Dict[str, Dict[str, float]]


def _extract_runs_qrels(obj):
    """
    Robustly unpack (runs, qrels) from common shapes produced by GridSearchCV splits.
    Accepts:
      - {"runs": [...], "qrels": qrels}
      - ([...runs...], qrels)
      - [ {"runs": [...], "qrels": qrels} ]
      - [([...runs...], qrels)]
    """
    runs = qrels = None

    if isinstance(obj, dict):
        runs = obj.get("runs")
        qrels = obj.get("qrels")
        return runs, qrels

    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and not isinstance(obj[0], dict):
            runs, qrels = obj
            return runs, qrels
        if len(obj) == 1:
            inner = obj[0]
            if isinstance(inner, dict):
                runs = inner.get("runs")
                qrels = inner.get("qrels")
                return runs, qrels
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                runs, qrels = inner
                return runs, qrels

    return runs, qrels


def weight_grid(
    num_runs: int,
    start: float = 0.1,
    end: float = 1.0,
    step: float = 0.1,
) -> list[list[float]]:
    """
    Generate all weight combinations for `num_runs`, stepping from start to end (inclusive).

    Example (num_runs=3, step=0.1) -> 1000 combinations for grid search.
    """
    if num_runs <= 0:
        return []
    if step <= 0 or end < start:
        raise ValueError("step must be > 0 and end >= start")

    n_steps = int(round((end - start) / step)) + 1
    values = [round(start + i * step, 10) for i in range(n_steps)]

    # Cartesian product without pulling in numpy
    combos: list[list[float]] = [[]]
    for _ in range(num_runs):
        combos = [c + [v] for c in combos for v in values]
    return combos


def reciprocal_rank_fusion(
    runs: Sequence[Mapping[str, Mapping[str, float]]],
    weights: Sequence[float] | None = None,
    k: float = 60.0,
    top_k: int | None = None,
    show_progress: bool = False,
) -> ResultDict:
    """
    Apply Reciprocal Rank Fusion over several ranked lists.

    Args:
        runs: Sequence of result dicts (qid -> {doc_id: score}) produced beforehand.
        weights: Optional gains per run. Defaults to 1.0 for each supplied run.
        k: RRF constant; larger values flatten the contribution of deeper ranks.
        top_k: Optional cap on how many fused docs to keep per query.
        show_progress: If True, display a per-query tqdm bar.
    """
    if not runs:
        return {}

    if weights is None:
        weights = [1.0] * len(runs)

    if len(weights) != len(runs):
        raise ValueError("weights must have the same length as runs")

    if k <= 0:
        raise ValueError("k must be > 0 to avoid division by zero")

    all_qids = set()
    for run in runs:
        all_qids.update(run.keys())

    q_iter = tqdm(
        list(all_qids),
        desc="RRF fuse",
        disable=not show_progress,
        mininterval=0.1,
        miniters=1,
        dynamic_ncols=True,
        leave=True,
    )
    fused: ResultDict = {}
    for qid in q_iter:
        agg_scores: Dict[str, float] = defaultdict(float)

        for run, gain in zip(runs, weights):
            q_run = run.get(qid, {})
            if not q_run:
                continue

            ranked = sorted(q_run.items(), key=lambda x: (-x[1], x[0]))
            for rank_idx, (doc_id, _) in enumerate(ranked, start=1):
                agg_scores[doc_id] += gain / (k + rank_idx)

        if not agg_scores:
            fused[qid] = {}
            continue

        sorted_docs = sorted(agg_scores.items(), key=lambda x: (-x[1], x[0]))
        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]

        fused[qid] = {doc_id: score for doc_id, score in sorted_docs}

    return fused


class RRFSklearnEstimator(BaseEstimator):
    """
    Sklearn-compatible estimator for tuning RRF hyperparameters (k, weights, top_k).

    Usage with GridSearchCV / HalvingGridSearchCV:
        est = RRFSklearnEstimator(k_eval=10)
        grid = {"k": [20, 60, 120], "weights": [[1,1,1], [0.5,1,0.8]]}
        search.fit({"runs": [bm25, dense, sparse], "qrels": qrels})
    """

    def __init__(
        self,
        k: float = 60.0,
        weights: Sequence[float] | None = None,
        top_k: int | None = None,
        k_eval: int = 10,
        show_progress: bool = False,
    ):
        self.k = k
        self.weights = weights
        self.top_k = top_k
        self.k_eval = k_eval
        self.show_progress = show_progress

    def fit(self, X, y=None):
        """
        X can be:
          - {"runs": [run1, run2, ...], "qrels": qrels}
          - ([run1, run2, ...], qrels)
        """
        runs, qrels = _extract_runs_qrels(X)

        if not runs:
            raise ValueError("RRFSklearnEstimator.fit expects non-empty runs.")
        if qrels is None:
            raise ValueError("RRFSklearnEstimator.fit expects qrels.")

        self.runs_ = list(runs)
        self.qrels_ = qrels
        return self

    def score(self, X=None, y=None):
        if not hasattr(self, "runs_") or not hasattr(self, "qrels_"):
            raise ValueError("Call fit with runs and qrels before scoring.")

        fused = reciprocal_rank_fusion(
            runs=self.runs_,
            weights=self.weights,
            k=self.k,
            top_k=self.top_k,
            show_progress=self.show_progress,
        )
        ndcg, *_ = EvaluateRetrievalCI.evaluate(
            self.qrels_,
            fused,
            k_values=[self.k_eval],
        )
        return ndcg[f"NDCG@{self.k_eval}"]["mean"]
