from __future__ import annotations

import glob
import logging
import math
import copy
from typing import Dict, List, Tuple

import numpy as np
import pytrec_eval

from beir.retrieval.custom_metrics import hole, mrr, recall_cap, top_k_accuracy
from beir.retrieval.search.base import BaseSearch
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


def _agg_stats(values: np.ndarray, z: float = 1.96) -> Tuple[float, float, Tuple[float, float]]:
    """
    Given per-query metric values, compute:
      - mean
      - sample variance
      - (low, high) 95% CI for the mean (normal approximation).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)

    mean = float(values.mean())
    if n > 1:
        var = float(values.var(ddof=1))
        se = math.sqrt(var / n)
        low = mean - z * se
        high = mean + z * se
    else:
        # With only one query, variance and CI are degenerate
        var = 0.0
        low = high = mean

    return mean, var, (low, high)


class EvaluateRetrievalCI:
    """
    Like EvaluateRetrieval, but evaluate() returns:
      ndcg['NDCG@10'] = {
          'mean': float,
          'var': float,
          'ci': (low, high)
      }
    (same for MAP, Recall, Precision).
    """

    def __init__(
        self,
        retriever: BaseSearch | None = None,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        score_function: str | None = "cos_sim",
    ):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)

    def encode_and_retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        self.retriever.encode(
            corpus,
            queries,
            encode_output_path=encode_output_path,
            overwrite=overwrite,
            query_filename=query_filename,
            corpus_filename=corpus_filename,
            **kwargs,
        )

        query_embeddings_file = f"{encode_output_path}/{query_filename}"
        corpus_embeddings_files = glob.glob(f"{encode_output_path}/{corpus_filename}")

        return self.retriever.search_from_files(
            query_embeddings_file=query_embeddings_file,
            corpus_embeddings_files=corpus_embeddings_files,
            top_k=self.top_k,
            **kwargs,
        )
    
    @staticmethod
    def _per_query_mrr_at_k(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Returns an array of per-query reciprocal ranks at cutoff k.
        If a query has no relevant document in the top-k, its RR is 0.
        """
        rrs: List[float] = []

        it = qrels.items()
        if show_progress:
            it = tqdm(it, total=len(qrels), desc=f"MRR@{k}", leave=False)

        for qid, rel_docs in it:
            # Relevant docs: those with relevance > 0
            relevant = {doc_id for doc_id, rel in rel_docs.items() if rel > 0}

            if qid not in results or len(results[qid]) == 0:
                rrs.append(0.0)
                continue

            # results[qid] is a dict[doc_id] -> score; sort by score desc, take top-k
            ranked_docs = sorted(
                results[qid].items(), key=lambda x: x[1], reverse=True
            )[:k]

            rr = 0.0
            for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break

            rrs.append(rr)

        return np.array(rrs, dtype=np.float64)

    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
        alpha: float = 0.05,
        show_progress: bool = False,
    ) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict], Dict[str, dict]]:

        """
        Returns four dicts. For example:

            ndcg['NDCG@10'] = {
                'mean': 0.48,
                'var': 0.01,
                'ci': (0.46, 0.50)
            }

        Same structure for MAP, Recall, Precision.
        """
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), "
                "please explicitly set ``ignore_identical_ids=False`` to keep this behaviour."
            )
            results = copy.deepcopy(results)
            it = results.items()
            if show_progress:
                it = tqdm(it, total=len(results), desc="filter ids", leave=False)
            for qid, rels in it:
                for pid in list(rels):
                    if qid == pid:
                        del results[qid][pid]

        # Prepare metric strings for pytrec_eval
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {map_string, ndcg_string, recall_string, precision_string},
        )
        scores = evaluator.evaluate(results)  # dict[qid] -> metric dict

        ndcg: Dict[str, dict] = {}
        _map: Dict[str, dict] = {}
        recall: Dict[str, dict] = {}
        precision: Dict[str, dict] = {}
        mrr_stats: Dict[str, dict] = {}


        z = 1.96  # 95% CI

        for k in k_values:
            # Collect per-query values for each metric
            ndcg_vals = np.array([s[f"ndcg_cut_{k}"] for s in scores.values()])
            map_vals = np.array([s[f"map_cut_{k}"] for s in scores.values()])
            recall_vals = np.array([s[f"recall_{k}"] for s in scores.values()])
            prec_vals = np.array([s[f"P_{k}"] for s in scores.values()])

            ndcg_mean, ndcg_var, ndcg_ci = _agg_stats(ndcg_vals, z=z)
            map_mean, map_var, map_ci = _agg_stats(map_vals, z=z)
            rec_mean, rec_var, rec_ci = _agg_stats(recall_vals, z=z)
            prec_mean, prec_var, prec_ci = _agg_stats(prec_vals, z=z)

            mrr_vals = EvaluateRetrievalCI._per_query_mrr_at_k(qrels, results, k, show_progress=show_progress)
            mrr_mean, mrr_var, mrr_ci = _agg_stats(mrr_vals, z=z)

            ndcg[f"NDCG@{k}"] = {"mean": ndcg_mean, "var": ndcg_var, "ci": ndcg_ci}
            _map[f"MAP@{k}"] = {"mean": map_mean, "var": map_var, "ci": map_ci}
            recall[f"Recall@{k}"] = {"mean": rec_mean, "var": rec_var, "ci": rec_ci}
            precision[f"P@{k}"] = {"mean": prec_mean, "var": prec_var, "ci": prec_ci}
            mrr_stats[f"MRR@{k}"] = {"mean": mrr_mean, "var": mrr_var, "ci": mrr_ci}


        # Logging
        for metric_name, metric_dict in [
            ("NDCG", ndcg),
            ("MAP", _map),
            ("Recall", recall),
            ("Precision", precision),
            ("MRR", mrr_stats),
        ]:
            logger.info("\n" + metric_name)
            for k, stats in metric_dict.items():
                logger.info(
                    f"{k}: mean={stats['mean']:.5f}, "
                    f"var={stats['var']:.5f}, "
                    f"ci=({stats['ci'][0]:.5f}, {stats['ci'][1]:.5f})"
                )


        return ndcg, _map, recall, precision, mrr_stats


    @staticmethod
    def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric: str,
    ):
        """
        You will need to adapt your custom metric functions so that they
        can return per-query values if you also want CI/variance for them.
        For now this just passes through to the original implementations.
        """
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)
