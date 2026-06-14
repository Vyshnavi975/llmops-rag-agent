"""Evaluation metrics, implemented from scratch (no external eval library).

Three complementary scores, each cheap and fully deterministic:

* `token_overlap_f1` — precision/recall/F1 over bag-of-words token sets
  between a prediction and a reference answer. Order-insensitive; rewards
  covering the reference's vocabulary without being a strict string match.
* `rouge_l_f1` — classic ROUGE-L, based on the Longest Common Subsequence
  (LCS) between the token sequences. Unlike bag-of-words overlap it is
  sensitive to word order, which token-overlap alone is not.
* `keyword_recall` — fraction of a hand-labeled list of "key facts" (short
  strings like "15%" or "SOC 2 Type II") that appear verbatim (case
  insensitive) in the prediction. This is the metric that most directly
  answers "did the answer contain the specific facts we care about?",
  which is often what actually matters for a support/RAG assistant, more
  than surface similarity to one reference phrasing.

None of these require an LLM-as-judge call, which matters for a demo-mode
eval harness that must run with no API key and no network access — and,
even in production, keeps the eval loop fast and free to run on every
prompt or model change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def token_overlap_f1(prediction: str, reference: str) -> dict:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts: dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = sum(min(pred_counts[t], ref_counts.get(t, 0)) for t in pred_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Standard O(len(a) * len(b)) dynamic-programming LCS length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l_f1(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure between prediction and reference token sequences,
    following Lin (2004): R_lcs = LCS/len(ref), P_lcs = LCS/len(pred),
    F_lcs = ((1+beta^2) * R * P) / (R + beta^2 * P).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    denom = recall + (beta ** 2) * precision
    if denom == 0:
        return 0.0
    return ((1 + beta ** 2) * recall * precision) / denom


def keyword_recall(prediction: str, key_facts: List[str]) -> dict:
    """Fraction of `key_facts` found verbatim (case-insensitive substring
    match) in `prediction`, plus which ones were found/missed for a
    human-readable report."""
    if not key_facts:
        return {"recall": 1.0, "found": [], "missed": []}
    pred_lower = prediction.lower()
    found = [fact for fact in key_facts if fact.lower() in pred_lower]
    missed = [fact for fact in key_facts if fact not in found]
    return {"recall": len(found) / len(key_facts), "found": found, "missed": missed}


@dataclass
class ExampleScore:
    example_id: str
    question: str
    prediction: str
    reference: str
    key_facts: List[str]
    f1: float
    rouge_l: float
    keyword_recall: float
    keyword_found: List[str]
    keyword_missed: List[str]
    passed: bool

    def to_dict(self) -> dict:
        return {
            "id": self.example_id,
            "question": self.question,
            "prediction": self.prediction,
            "reference": self.reference,
            "f1": round(self.f1, 3),
            "rouge_l": round(self.rouge_l, 3),
            "keyword_recall": round(self.keyword_recall, 3),
            "keyword_found": self.keyword_found,
            "keyword_missed": self.keyword_missed,
            "passed": self.passed,
        }


def score_example(
    example_id: str,
    question: str,
    prediction: str,
    reference: str,
    key_facts: List[str],
    pass_threshold: float = 0.5,
) -> ExampleScore:
    """Score one prediction against one labeled example. An example
    "passes" when keyword recall (did it contain the facts we care about)
    is at or above `pass_threshold` — this is deliberately the primary
    pass/fail signal, since for a support-style RAG assistant getting the
    specific facts right matters more than matching one reference's exact
    phrasing.
    """
    f1_scores = token_overlap_f1(prediction, reference)
    rouge = rouge_l_f1(prediction, reference)
    kw = keyword_recall(prediction, key_facts)
    passed = kw["recall"] >= pass_threshold
    return ExampleScore(
        example_id=example_id,
        question=question,
        prediction=prediction,
        reference=reference,
        key_facts=key_facts,
        f1=f1_scores["f1"],
        rouge_l=rouge,
        keyword_recall=kw["recall"],
        keyword_found=kw["found"],
        keyword_missed=kw["missed"],
        passed=passed,
    )
