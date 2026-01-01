# BEIRag

End-to-end retrieval and grounded QA workflow on the BEIR MS MARCO (dev) dataset,
covering lexical, sparse, dense, fusion, reranking, and answer generation with
evaluation utilities.

## Highlights
- Notebook-first pipeline in `BEIRag.ipynb` (retrieval to QA and evaluation).
- Retrieval wrappers in `models/` (BM25s, Pyserini BM25+RM3, SPLADE sparse, RRF, Qwen).
- Evaluation helpers in `utils/` (confidence intervals, plotting, OpenAI grading).
- Cached result runs in `results_cache/`.

## Quickstart
1. Open `BEIRag.ipynb` in Jupyter/Colab.
2. Install dependencies using the notebook's install cell.
3. Run cells in order to download data, build indexes/embeddings, and evaluate.

Optional: the OpenAI evaluator expects Azure OpenAI credentials:

```bash
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_VERSION=2024-12-01-preview
export EVALUATION_MODEL_NAME=...
```

## Repository layout
- `BEIRag.ipynb`: main end-to-end experiment notebook.
- `models/`: retrieval, fusion, and generation wrappers.
- `utils/`: evaluation, plotting, and OpenAI grading helpers.
- `results_cache/`: cached retrieval runs (LFS).
- `data/`, `shards/`, `embeddings_bge_m3/`: dataset artifacts and embeddings.
- `api_responses/`: stored OpenAI grading responses (if used).

## Notes
- Pyserini BM25+RM3 requires Java and a working Pyserini installation. (Not actively used in our scenario though)
- Large binary artifacts are stored with Git LFS.
