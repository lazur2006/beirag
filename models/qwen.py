from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


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


def load_causal_lm(
    model_name: str,
    *,
    device_map: Union[str, Dict[str, int]] = "auto",
    torch_dtype: Union[str, torch.dtype] = "auto",
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
):

    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def generate_chat_with_thinking(
    model,
    tokenizer,
    *,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 1024,
    enable_thinking: bool = True,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:

    if messages is None:
        if prompt is None:
            raise ValueError("Provide either `prompt` or `messages`.")
        messages = [{"role": "user", "content": prompt}]

    generation_kwargs = generation_kwargs or {}

    # Build model input using the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
            **generation_kwargs,
        )

    # Only keep newly generated tokens (post prompt)
    input_len = model_inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][input_len:].tolist()
    return _decode_generated_output(tokenizer, output_ids)


def _decode_generated_output(tokenizer, output_ids: List[int]) -> Dict[str, str]:
    """Decode a single generation and split thinking vs final content (token-id based, matches HF example)."""
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    split_idx = 0
    if think_end_id is not None and think_end_id != getattr(tokenizer, "unk_token_id", None):
        try:
            split_idx = len(output_ids) - output_ids[::-1].index(think_end_id)
        except ValueError:
            split_idx = 0

    thinking = tokenizer.decode(output_ids[:split_idx], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[split_idx:], skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    # If no closing tag but an opening exists, treat all text as both thinking and content to avoid empty answers.
    if split_idx == 0 and "<think>" in raw_text and "</think>" not in raw_text:
        merged = thinking or content
        return {"thinking": merged, "content": merged, "raw_text": raw_text}

    if not content and thinking:
        content = thinking
    return {"thinking": thinking, "content": content, "raw_text": raw_text}


def batch_generate_chat_with_thinking(
    model,
    tokenizer,
    prompts: List[str],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    max_length: int = 2048,
    enable_thinking: bool = True,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    use_bfloat16: bool = True,
    show_progress: bool = False,
    strict_single: bool = False,
) -> List[Dict[str, str]]:
    """
    Batched chat generation with optional thinking tokens and autocast for speed.

    If `strict_single=True`, fall back to calling `generate_chat_with_thinking`
    one-by-one (useful when using sampling and you need identical outputs to the
    single-call helper).
    Returns a list of dicts matching `generate_chat_with_thinking` output.
    """

    generation_kwargs = generation_kwargs or {}
    results: List[Dict[str, str]] = []
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bfloat16 and model.device.type == "cuda"
        else contextlib.nullcontext()
    )

    if strict_single:
        progress = _maybe_progress(len(prompts), "Chat generation (strict)") if show_progress else None
        for prompt in prompts:
            results.append(
                generate_chat_with_thinking(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    generation_kwargs=generation_kwargs,
                )
            )
            if progress:
                progress.update(1)
        if progress:
            progress.close()
        return results

    progress = _maybe_progress(len(prompts), "Chat generation") if show_progress else None

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for prompt in chunk
        ]

        inputs = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        with torch.no_grad(), autocast_context:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                **generation_kwargs,
            )

        gen_start = inputs["input_ids"].shape[1]  # start of newly generated tokens for left-padded batches
        for seq in generated:
            output_ids = seq[gen_start:].tolist()
            results.append(_decode_generated_output(tokenizer, output_ids))
            if progress:
                progress.update(1)

    if progress:
        progress.close()
    return results
