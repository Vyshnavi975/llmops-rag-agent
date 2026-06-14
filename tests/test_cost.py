"""Unit tests for ops/cost.py: token counting and cost math.

`count_tokens` is tested against the word-based fallback approximation
directly (by forcing `_TIKTOKEN_AVAILABLE = False` via monkeypatch), since
that path's arithmetic is simple enough to hand-verify and must work
identically whether or not tiktoken happens to be installed in the test
environment. `estimate_cost_usd` and `CostTracker` are tested against the
real price table, which is itself a plain Python constant, so its values
are asserted directly.
"""

import ops.cost as cost_mod
from ops.cost import CostTracker, count_tokens, estimate_cost_usd


class TestApproxTokenFallback:
    def test_word_count_approximation_hand_computed(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        # 5 words * 1.3 = 6.5 -> int(6.5 + 0.5) = int(7.0) = 7
        text = "the quick brown fox jumps"
        assert count_tokens(text) == 7

    def test_single_word_rounds_up_to_at_least_one(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        assert count_tokens("hello") == max(1, int(1 * 1.3 + 0.5))
        assert count_tokens("hello") == 1

    def test_empty_text_is_zero_tokens(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        assert count_tokens("") == 0

    def test_ten_words_hand_computed(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        # 10 words * 1.3 = 13.0 -> int(13.0 + 0.5) = 13
        text = " ".join(["word"] * 10)
        assert count_tokens(text) == 13


class TestEstimateCostUsd:
    def test_known_model_price_hand_computed(self):
        # gpt-4o-mini: (0.000150, 0.000600) per 1K tokens
        # 1000 input tokens -> 0.000150 ; 500 output tokens -> 0.0003
        cost = estimate_cost_usd(1000, 500, model="gpt-4o-mini")
        assert abs(cost - (0.000150 + 0.0003)) < 1e-12

    def test_zero_tokens_costs_zero(self):
        assert estimate_cost_usd(0, 0, model="gpt-4o") == 0.0

    def test_demo_model_is_always_free(self):
        assert estimate_cost_usd(10_000, 10_000, model="demo-fake-llm") == 0.0

    def test_unknown_model_falls_back_to_default_pricing(self):
        default_price = cost_mod.PRICE_TABLE_PER_1K[cost_mod.DEFAULT_MODEL_FOR_UNKNOWN]
        expected = (1000 / 1000.0) * default_price[0] + (1000 / 1000.0) * default_price[1]
        assert estimate_cost_usd(1000, 1000, model="some-model-not-in-table") == expected


class TestCostTracker:
    def test_records_and_aggregates_multiple_calls(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        tracker = CostTracker(model="gpt-4o-mini")

        tracker.record(node="generate", prompt_text="a b c d e", completion_text="f g")
        tracker.record(node="generate", prompt_text="h i", completion_text="j k l m")

        # call 1: input "a b c d e" -> 5 words -> int(5*1.3+0.5)=7 ; output "f g" -> 2 words -> int(2*1.3+0.5)=3
        # call 2: input "h i" -> 2 words -> 3 ; output "j k l m" -> 4 words -> int(4*1.3+0.5)=5
        assert tracker.total_input_tokens == 7 + 3
        assert tracker.total_output_tokens == 3 + 5
        assert tracker.total_tokens == 7 + 3 + 3 + 5
        assert len(tracker.calls) == 2

        expected_cost = estimate_cost_usd(7, 3, "gpt-4o-mini") + estimate_cost_usd(3, 5, "gpt-4o-mini")
        assert abs(tracker.total_cost_usd - expected_cost) < 1e-12

    def test_summary_shape(self, monkeypatch):
        monkeypatch.setattr(cost_mod, "_TIKTOKEN_AVAILABLE", False)
        tracker = CostTracker(model="demo-fake-llm")
        tracker.record(node="generate", prompt_text="hello world", completion_text="hi")
        summary = tracker.summary()
        assert summary["num_llm_calls"] == 1
        assert summary["total_cost_usd"] == 0.0  # demo-fake-llm is free
        assert len(summary["calls"]) == 1
        assert summary["calls"][0]["node"] == "generate"

    def test_empty_tracker_has_zero_totals(self):
        tracker = CostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.summary()["num_llm_calls"] == 0
