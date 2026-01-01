"""
Learned sparse retrieval wrappers that avoid patching the installed BEIR package.
"""

from __future__ import annotations

import os
import pickle
import heapq
from typing import Dict, List

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers.util import batch_to_device
from torch import Tensor
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from scipy import sparse as sp
except ImportError:  # optional dependency
    sp = None

from tqdm.autonotebook import trange

from beir.retrieval.search.base import BaseSearch


def _extract_corpus_sentences(corpus: list[dict[str, str]] | dict[str, list] | list[str], sep: str) -> list[str]:
    """Minimal clone of beir.retrieval.models.util.extract_corpus_sentences to avoid importing beir.models.__init__."""
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]
    elif isinstance(corpus, list):
        if isinstance(corpus[0], str):
            sentences = corpus
        else:
            sentences = [
                (doc.get("title", "") + sep + doc.get("text", "")).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
    else:
        sentences = []
    return sentences


class _SpladeNaver(torch.nn.Module):
    """Copied from beir.retrieval.models.splade to keep behavior without importing the package-level __init__."""

    def __init__(self, model_path):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]
        return torch.max(
            torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values

    def _text_length(self, text: list[int] | list[list[int]]) -> int:
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])

    def encode_sentence_bert(
        self,
        tokenizer,
        sentences: str | list[str] | list[int],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        maxlen: int = 512,
        is_q: bool = False,
    ) -> list[Tensor] | ndarray | Tensor:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == "token_embeddings":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = tokenizer(
                sentences_batch,
                add_special_tokens=True,
                padding="longest",
                truncation="only_first",
                max_length=maxlen,
                return_attention_mask=True,
                return_tensors="pt",
            )
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(**features)
                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0 : last_mask_id + 1])
                else:
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings


class LSR:
    """Drop-in SPLADE using only code to sidestep beir.retrieval.models imports."""

    def __init__(self, model_path: str | None = None, sep: str = " ", max_length: int = 256, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = _SpladeNaver(model_path)
        self.sep = sep
        self.model.eval()

    def encode_queries(self, queries: list[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.model.encode_sentence_bert(self.tokenizer, queries, is_q=True, maxlen=self.max_length)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int, **kwargs
    ):
        if sp is None:
            raise ImportError("scipy is required for sparse SPLADE encoding (install scipy).")
        sentences = _extract_corpus_sentences(corpus=corpus, sep=self.sep)
        # Pull embeddings to CPU NumPy to avoid accumulating GPU tensors
        embeddings = self.model.encode_sentence_bert(
            self.tokenizer,
            sentences,
            batch_size=batch_size,
            maxlen=self.max_length,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        vocab_size = int(embeddings[0].shape[-1]) if len(embeddings) else 0
        for doc_idx, emb in enumerate(embeddings):
            # emb is a 1D numpy array on CPU
            nz = np.nonzero(emb)[0]
            if nz.size == 0:
                continue
            rows.extend(nz.tolist())
            cols.extend([doc_idx] * nz.size)
            data.extend(emb[nz].astype(np.float32).tolist())

        return sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, len(embeddings)), dtype=np.float32)

    def encode_query(self, query: str, batch_size: int = 1, **kwargs) -> np.ndarray:
        return self.encode_queries([query], batch_size=batch_size, **kwargs)[0]


class SparseSearchHF(BaseSearch):
    """Sparse search for SPLADE/UniCOIL-style models without touching site-packages."""

    def __init__(self, model, batch_size: int = 16):
        self.model = model
        self.batch_size = batch_size
        self.sparse_matrix = None
        self.results: Dict[str, Dict[str, float]] = {}

    def _encode_query(self, query: str):
        if hasattr(self.model, "encode_query"):
            return self.model.encode_query(query)
        if hasattr(self.model, "encode_queries"):
            return self.model.encode_queries([query], batch_size=1)[0]
        raise AttributeError("Model must implement encode_query or encode_queries.")

    def _is_weighted_query(self, query_tokens) -> bool:
        if sp is not None and sp.issparse(query_tokens):
            return True
        if isinstance(query_tokens, np.ndarray):
            return query_tokens.ndim > 0 and not np.issubdtype(query_tokens.dtype, np.integer)
        if isinstance(query_tokens, (list, tuple)) and query_tokens:
            return isinstance(query_tokens[0], float)
        return False

    def _score(self, query_tokens, doc_ids: List[str], query_weights: bool):
        if self.sparse_matrix is None:
            raise ValueError("Corpus embeddings not computed. Call encode_corpus before scoring.")

        use_weights = query_weights or self._is_weighted_query(query_tokens)
        matrix = self.sparse_matrix

        if use_weights:
            if sp is not None and sp.issparse(query_tokens):
                vector = query_tokens.A1
            else:
                vector = np.asarray(query_tokens).squeeze()

            if matrix.shape[1] == vector.shape[0]:
                scores = matrix.dot(vector)
            elif matrix.shape[0] == vector.shape[0]:
                scores = matrix.T.dot(vector)
            else:
                raise ValueError(
                    f"Query embedding dim ({vector.shape[0]}) does not align with corpus matrix shape {matrix.shape}"
                )
        else:
            token_ids = list(query_tokens)
            if matrix.shape[1] == len(doc_ids):
                scores = matrix[token_ids, :].sum(axis=0)
            elif matrix.shape[0] == len(doc_ids):
                scores = matrix[:, token_ids].sum(axis=1)
            else:
                raise ValueError(
                    f"Corpus matrix shape {matrix.shape} does not align with number of documents ({len(doc_ids)})"
                )

        if sp is not None and sp.issparse(scores):
            return scores.A1
        return np.asarray(scores).squeeze()

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str,
        query_weights: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        documents = [corpus[doc_id] for doc_id in doc_ids]
        self.sparse_matrix = self.model.encode_corpus(documents, batch_size=self.batch_size, **kwargs)
        self.results = {}

        for idx in trange(0, len(queries), desc="query"):
            qid = query_ids[idx]
            query_tokens = self._encode_query(queries[qid])
            scores = self._score(query_tokens, doc_ids, query_weights=query_weights)
            k = min(top_k, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            self.results[qid] = {doc_ids[i]: float(scores[i]) for i in top_idx if doc_ids[i] != qid}

        return self.results

    def encode(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        shard_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        os.makedirs(encode_output_path, exist_ok=True)

        # Queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        qpath = os.path.join(encode_output_path, query_filename)
        if overwrite or not os.path.exists(qpath):
            if hasattr(self.model, "encode_queries"):
                q_embs = self.model.encode_queries(query_texts, batch_size=self.batch_size, **kwargs)
            else:
                q_embs = [self.model.encode_query(q) for q in query_texts]
            with open(qpath, "wb") as f:
                pickle.dump((q_embs, query_ids), f)

        # Corpus encoding (optional sharding to keep memory small)
        corpus_ids = list(corpus.keys())
        documents = [corpus[doc_id] for doc_id in corpus_ids]
        if shard_size is None or shard_size <= 0:
            cpath = os.path.join(encode_output_path, corpus_filename.replace("*", "0"))
            if overwrite or not os.path.exists(cpath):
                c_embs = self.model.encode_corpus(documents, batch_size=self.batch_size, **kwargs)
                with open(cpath, "wb") as f:
                    pickle.dump((c_embs, corpus_ids), f)
        else:
            import math
            from scipy.sparse import save_npz

            num_shards = math.ceil(len(documents) / shard_size)
            shard_iter = range(num_shards)
            if show_progress:
                shard_iter = trange(num_shards, desc="shards", leave=True)
            for shard_idx in shard_iter:
                start = shard_idx * shard_size
                end = min(len(documents), start + shard_size)
                sub_ids = corpus_ids[start:end]
                sub_docs = documents[start:end]
                shard_path = os.path.join(encode_output_path, corpus_filename.replace("*", str(shard_idx)))
                ids_path = shard_path + ".ids"
                if overwrite or not os.path.exists(shard_path):
                    c_embs = self.model.encode_corpus(sub_docs, batch_size=self.batch_size, **kwargs)
                    save_npz(shard_path, c_embs)
                    with open(ids_path, "wb") as f:
                        pickle.dump(sub_ids, f)

    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: List[str],
        top_k: int,
        score_function: str = None,
        query_weights: bool = False,
        show_progress: bool = False,
        show_query_progress: bool = False,
        shard_batch_size: int = 1,
        query_batch_size: int | None = None,
        device: str | torch.device | None = None,
        merge_shard_batch: bool = False,
        dtype: torch.dtype | str = torch.float32,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if shard_batch_size <= 0:
            raise ValueError("shard_batch_size must be >= 1")
        if sp is None:
            raise ImportError("scipy is required for sparse search_from_files")

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Load all queries once, stack, and move to target device
        q_embs, q_ids = pickle.load(open(query_embeddings_file, "rb"))
        q_array = np.asarray(q_embs)
        if q_array.ndim == 1:
            q_array = q_array.reshape(1, -1)
        if query_batch_size is None or query_batch_size <= 0:
            query_batch_size = len(q_ids)

        resolved_device = device
        if resolved_device is None:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(resolved_device)

        q_tensor = torch.as_tensor(q_array, dtype=dtype, device=resolved_device)

        # per-query min-heap of (score, doc_id) to cap memory at top_k
        self.results = {qid: [] for qid in q_ids}

        shard_files = sorted(corpus_embeddings_files)
        shard_indices = range(0, len(shard_files), shard_batch_size)
        shard_iter = shard_indices
        if show_progress:
            shard_iter = trange(len(shard_indices), desc="score shards", leave=True)

        def _load_shard(path: str):
            if path.endswith(".npz"):
                from scipy.sparse import load_npz

                shard_matrix = load_npz(path)
                ids_path = path + ".ids"
                if os.path.exists(ids_path):
                    shard_ids = pickle.load(open(ids_path, "rb"))
                else:
                    raise FileNotFoundError(f"Missing ids file for shard: {ids_path}")
            else:
                # legacy pickle contains (matrix, ids)
                shard_matrix, shard_ids = pickle.load(open(path, "rb"))
                shard_ids = list(shard_ids)
            return shard_matrix.tocsr(), shard_ids

        for idx_offset in shard_iter:
            # When using trange, idx_offset is a position, so map it back to start index
            if show_progress:
                start = idx_offset * shard_batch_size
            else:
                start = idx_offset
            shard_batch_paths = shard_files[start : start + shard_batch_size]
            shard_batch = [_load_shard(p) for p in shard_batch_paths]

            # Optionally merge the loaded shards into a single block to reduce per-shard overhead
            if merge_shard_batch and len(shard_batch) > 1:
                shard_matrices, shard_ids = zip(*shard_batch)
                merged_matrix = sp.hstack(shard_matrices, format="csr")
                merged_ids: list[str] = []
                for ids in shard_ids:
                    merged_ids.extend(ids)
                shard_batch = [(merged_matrix, merged_ids)]

            for shard_matrix, doc_ids in shard_batch:
                # Build torch sparse CSR on target device
                crow_indices = torch.as_tensor(shard_matrix.indptr, device=resolved_device)
                col_indices = torch.as_tensor(shard_matrix.indices, device=resolved_device)
                values = torch.as_tensor(shard_matrix.data, dtype=dtype, device=resolved_device)
                sp_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=shard_matrix.shape, device=resolved_device)
                sp_tensor_t = sp_tensor.transpose(0, 1)

                if show_query_progress:
                    q_iter = trange(0, len(q_ids), query_batch_size, desc="score queries", leave=False)
                else:
                    q_iter = range(0, len(q_ids), query_batch_size)

                for q_start in q_iter:
                    q_end = min(len(q_ids), q_start + query_batch_size)
                    q_batch = q_tensor[q_start:q_end]  # (batch, dim)
                    if q_batch.numel() == 0:
                        continue

                    # docs x batch scores
                    scores = torch.matmul(sp_tensor_t, q_batch.T)
                    k = min(top_k, scores.shape[0])
                    if k == 0:
                        continue

                    top_scores, top_indices = torch.topk(scores, k=k, dim=0)
                    # bring small tensors back to CPU for heap operations
                    top_scores = top_scores.transpose(0, 1).contiguous().cpu().numpy()
                    top_indices = top_indices.transpose(0, 1).contiguous().cpu().numpy()

                    for row_idx, q_global_idx in enumerate(range(q_start, q_end)):
                        qid = q_ids[q_global_idx]
                        heap = self.results[qid]
                        for j in range(k):
                            doc_idx = int(top_indices[row_idx, j])
                            doc_id = doc_ids[doc_idx]
                            if doc_id == qid:
                                continue
                            score = float(top_scores[row_idx, j])
                            if len(heap) < top_k:
                                heapq.heappush(heap, (score, doc_id))
                            elif score > heap[0][0]:
                                heapq.heapreplace(heap, (score, doc_id))

            # free loaded shard batch explicitly
            shard_batch.clear()

        for qid, heap in list(self.results.items()):
            # heap holds (score, doc_id); sort descending
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
            self.results[qid] = {doc_id: score for score, doc_id in sorted_items}

        return self.results
