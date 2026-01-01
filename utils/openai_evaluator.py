"""
Standalone helper for scoring a single prediction against a ground-truth answer.

Input format per example row: [query, gold_answer, prediction].
Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in the environment before
calling `evaluate_prediction`. Optionally override AZURE_OPENAI_API_VERSION and
EVALUATION_MODEL_NAME (deployment name) if needed.
"""

import json
import os
import re
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from openai import APIConnectionError, AzureOpenAI, RateLimitError
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

DEFAULT_SYSTEM_MESSAGE = (
    "You are an assistant that grades answers.\n"
    "Given a question, a ground truth answer, and a prediction, decide if the "
    "prediction is correct. Reply with JSON: "
    '{"score": 1 or 0, "explanation": "brief reason"}.'
)

load_dotenv()

DEFAULT_EVAL_MODEL = (os.getenv("EVALUATION_MODEL_NAME"))
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
MAX_RETRIES = 10
MAX_TOKEN_LENGTH = 75


class _SimpleProgress:
    """Lightweight progress reporter if tqdm is unavailable."""

    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.current = 0
        print(f"{self.desc}: 0/{self.total}")

    def update(self, n: int):
        self.current += n
        shown = self.current if self.current <= self.total else self.total
        print(f"{self.desc}: {shown}/{self.total}")

    def close(self):
        return None


def _maybe_progress(total: int, desc: str):
    if not total:
        return None
    if tqdm is not None:
        return tqdm(total=total, desc=desc)
    return _SimpleProgress(total, desc)


def get_system_message() -> str:
    """Return the grading instruction for the evaluator."""
    return DEFAULT_SYSTEM_MESSAGE


def _build_default_client() -> AzureOpenAI:
    """Construct an Azure OpenAI client from environment variables."""
    missing = []
    if not AZURE_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI settings: {', '.join(missing)}")

    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


def attempt_api_call(client: AzureOpenAI, model_name: str, messages: List[dict]) -> str:
    """Call OpenAI chat completions with retries for rate/connection errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            # Retry on transient failures.
            continue
        except Exception:
            break
    return ""


def log_response(messages: List[dict], response: str, output_directory: str = "api_responses") -> None:
    """Persist the OpenAI response for traceability."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str) -> Tuple[str, int]:
    """
    Extract (explanation, score) from the OpenAI response text.
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score not in (0, 1):
                raise ValueError("bad score")
        else:
            return "Parse Err: Score not found", -1

        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        return text, score
    except Exception:
        return response, -1


def trim_prediction(prediction: str) -> str:
    """Trim the prediction to roughly 75 tokens using whitespace tokenization."""
    tokens = str(prediction or "").split()
    return " ".join(tokens[:MAX_TOKEN_LENGTH])


def _to_ground_truth_list(ground_truth: Iterable) -> List[str]:
    """Normalize ground truth into a list of strings."""
    if isinstance(ground_truth, str):
        return [ground_truth.strip()]
    if isinstance(ground_truth, Iterable):
        return [str(item).strip() for item in ground_truth]
    return [str(ground_truth).strip()]


def _evaluate_one_prediction(
    query: str,
    ground_truths: Sequence[str],
    prediction: str,
    client: AzureOpenAI,
    system_message: str,
    evaluation_model_name: str,
    log_dir: str,
) -> Tuple[bool, bool]:
    """
    Evaluate a single prediction against a list of ground truths.

    Returns (is_correct, is_missing) where missing captures "I don't know".
    """
    cleaned_prediction = trim_prediction(str(prediction or "").strip())
    prediction_lower = cleaned_prediction.lower()
    if "i don't know" in prediction_lower:
        return False, True

    accuracy = -1
    for ground_truth in ground_truths:
        ground_truth_lower = ground_truth.lower()
        if prediction_lower == ground_truth_lower:
            accuracy = 1
            break
        if "invalid" in prediction_lower and "invalid" in ground_truth_lower:
            accuracy = 1
            break
        if "invalid" in prediction_lower and "invalid" not in ground_truth_lower:
            accuracy = 0
            continue
        if "invalid" not in prediction_lower and "invalid" in ground_truth_lower:
            accuracy = 0
            continue

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {cleaned_prediction}\n",
            },
        ]
        response = attempt_api_call(client, evaluation_model_name, messages)
        if response:
            log_response(messages, response, output_directory=log_dir)
            _, accuracy = parse_response(response)
            if accuracy == 1:
                break

    return accuracy == 1, False


def _finalize_metrics(counter: dict) -> dict:
    """Compute derived metrics for a bucket with counts."""
    n = counter["total"]
    n_correct = counter["n_correct"]
    n_miss = counter["n_miss"]
    n_hallucination = n - n_correct - n_miss

    counter.update(
        {
            "n_hallucination": n_hallucination,
            "accuracy": n_correct / n if n else 0.0,
            "missing": n_miss / n if n else 0.0,
            "hallucination": n_hallucination / n if n else 0.0,
            "score": (2 * n_correct + n_miss) / n - 1 if n else 0.0,
        }
    )
    return counter


def finalize_metrics(counter: dict) -> dict:
    """Public wrapper to derive aggregate metrics from accumulated counts."""
    return _finalize_metrics(counter)


def evaluate_prediction(
    query: str,
    gold: Sequence,
    prediction: str,
    *,
    evaluation_model_name: str = DEFAULT_EVAL_MODEL,
    log_dir: str = "api_responses",
    bucket: Optional[dict] = None,
    client: Optional[AzureOpenAI] = None,
) -> Tuple[dict, dict]:
    """
    Evaluate a single prediction against a query + ground truth.

    Returns (bucket, result) where:
      - bucket keeps running counts: {"total", "n_correct", "n_miss"}
      - result is {"is_correct": bool, "is_missing": bool}
    """
    bucket = bucket or {"total": 0, "n_correct": 0, "n_miss": 0}
    client = client or _build_default_client()
    system_message = get_system_message()

    ground_truths = _to_ground_truth_list(gold)
    bucket["total"] += 1

    is_correct, is_missing = _evaluate_one_prediction(
        query,
        ground_truths,
        prediction,
        client,
        system_message,
        evaluation_model_name,
        log_dir,
    )

    if is_missing:
        bucket["n_miss"] += 1
    elif is_correct:
        bucket["n_correct"] += 1

    return bucket, {"is_correct": is_correct, "is_missing": is_missing}


def evaluate_predictions_batch(
    items: Sequence[Tuple[str, Sequence[str], str]],
    *,
    evaluation_model_name: str = DEFAULT_EVAL_MODEL,
    log_dir: str = "api_responses",
    max_workers: int = 8,
    show_progress: bool = False,
    client: Optional[AzureOpenAI] = None,
    system_message: Optional[str] = None,
) -> Tuple[List[Tuple[bool, bool]], dict]:
    """
    Evaluate many predictions concurrently with optional progress reporting.

    Args:
        items: Sequence of (query, ground_truths, prediction).
        evaluation_model_name: Target Azure OpenAI deployment name.
        log_dir: Directory to store API responses.
        max_workers: Number of threads for parallel grading.
        show_progress: Enable a progress bar/printout.
        client: Optional AzureOpenAI client; built from env vars if omitted.
        system_message: Optional override for grading instructions.

    Returns:
        results: List of (is_correct, is_missing) in the same order as `items`.
        metrics: Derived metrics including counts via `finalize_metrics`.
    """

    total = len(items)
    client = client or _build_default_client()
    system_message = system_message or get_system_message()

    bucket = {"total": 0, "n_correct": 0, "n_miss": 0}
    results: List[Tuple[bool, bool]] = [(False, False)] * total
    progress = _maybe_progress(total, "OpenAI grading") if show_progress else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, (query, ground_truths, prediction) in enumerate(items):
            gt_list = _to_ground_truth_list(ground_truths)
            future = executor.submit(
                _evaluate_one_prediction,
                query,
                gt_list,
                prediction,
                client,
                system_message,
                evaluation_model_name,
                log_dir,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                is_correct, is_missing = future.result()
            except Exception:
                is_correct, is_missing = False, False
            results[idx] = (is_correct, is_missing)
            bucket["total"] += 1
            if is_missing:
                bucket["n_miss"] += 1
            elif is_correct:
                bucket["n_correct"] += 1
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    return results, finalize_metrics(bucket.copy())


if __name__ == "__main__":
    # Example usage. Replace `example_row` with your own data before running.
    example_row = [
        "Who wrote The Taming of the Shrew?",
        "william shakespeare",
        "William Shakespeare",
    ]
    bucket, result = evaluate_prediction(*example_row)
    print("Evaluation result:", result)
    print("Aggregated metrics:", finalize_metrics(bucket.copy()))
