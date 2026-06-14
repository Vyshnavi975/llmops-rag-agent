"""Token counting and cost estimation.

Every real LLM call in production has a dollar cost attached to it, and that
cost is invisible unless something measures it. This module counts tokens
per call (via `tiktoken` when available, falling back to a documented
word-based approximation otherwise) and prices them against a small,
explicit table of per-model rates, so a `RunTracer` (see `ops/tracing.py`)
can attach a `tokens_in`, `tokens_out`, and `cost_usd` figure to every node
of an agent run and roll them up into a per-run total.

Prices are USD per 1,000 tokens and are illustrative, publicly-known
ballpark figures for common OpenAI models at the time this project was
written — not pulled live from any billing API. In a real production
system this table would be a config value updated when providers change
pricing, not a constant baked into source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if tiktoken is absent
    _TIKTOKEN_AVAILABLE = False

# USD per 1,000 tokens: (input_price, output_price)
PRICE_TABLE_PER_1K: Dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0050, 0.0150),
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4-turbo": (0.0100, 0.0300),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
    # The offline demo "model" is, by construction, free.
    "demo-fake-llm": (0.0, 0.0),
}

DEFAULT_MODEL_FOR_UNKNOWN = "gpt-4o-mini"


def _approx_tokens_by_words(text: str) -> int:
    """Fallback token estimate when tiktoken isn't installed: OpenAI-style
    tokenizers average roughly 1.3 tokens per whitespace-delimited English
    word, so this multiplies word count by 1.3 and rounds up. It is a
    documented approximation, not a precise count.
    """
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3 + 0.5))


_encoding_cache: Dict[str, "tiktoken.Encoding"] = {}
_tiktoken_unavailable_at_runtime = False


def count_tokens(text: str, model: str = DEFAULT_MODEL_FOR_UNKNOWN) -> int:
    """Count tokens in `text` for `model`. Uses tiktoken's encoding for the
    model when tiktoken is installed (falling back to the general-purpose
    `cl100k_base` encoding for unrecognized model names), otherwise uses a
    word-count-based approximation."""
    global _tiktoken_unavailable_at_runtime
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and not _tiktoken_unavailable_at_runtime:
        try:
            if model not in _encoding_cache:
                try:
                    _encoding_cache[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
            return len(_encoding_cache[model].encode(text))
        except Exception:
            # tiktoken's BPE ranks are fetched from a remote blob store on
            # first use and cached locally on disk; in a fully offline /
            # air-gapped environment (or before that first fetch has ever
            # succeeded) this raises a network error rather than KeyError.
            # Fall back to the word-based approximation for the rest of the
            # process rather than retrying a doomed network call on every
            # single LLM call, or letting token/cost tracking take down the
            # agent run.
            _tiktoken_unavailable_at_runtime = True
            return _approx_tokens_by_words(text)
    return _approx_tokens_by_words(text)


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str = DEFAULT_MODEL_FOR_UNKNOWN) -> float:
    input_price, output_price = PRICE_TABLE_PER_1K.get(
        model, PRICE_TABLE_PER_1K[DEFAULT_MODEL_FOR_UNKNOWN]
    )
    return (input_tokens / 1000.0) * input_price + (output_tokens / 1000.0) * output_price


@dataclass
class CallCost:
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates `CallCost` records for a single agent run and rolls them
    up into totals. One instance is created per run by the agent graph."""

    model: str = DEFAULT_MODEL_FOR_UNKNOWN
    calls: List[CallCost] = field(default_factory=list)

    def record(self, node: str, prompt_text: str, completion_text: str, model: Optional[str] = None) -> CallCost:
        model = model or self.model
        input_tokens = count_tokens(prompt_text, model)
        output_tokens = count_tokens(completion_text, model)
        cost = estimate_cost_usd(input_tokens, output_tokens, model)
        call = CallCost(node=node, model=model, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
        self.calls.append(call)
        return call

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def summary(self) -> dict:
        return {
            "num_llm_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "calls": [
                {
                    "node": c.node,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": round(c.cost_usd, 6),
                }
                for c in self.calls
            ],
        }
