"""Unit tests for ops/metrics.py, with hand-verifiable expected values.

None of these require an API key, network access, or the agent graph —
they exercise the scoring functions directly against fixed strings.
"""

from ops.metrics import keyword_recall, rouge_l_f1, score_example, token_overlap_f1


class TestTokenOverlapF1:
    def test_identical_strings_score_perfect(self):
        result = token_overlap_f1("the cat sat on the mat", "the cat sat on the mat")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_completely_disjoint_strings_score_zero(self):
        result = token_overlap_f1("apples oranges bananas", "trucks cars planes")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_hand_computed_partial_overlap(self):
        # prediction tokens: {the, cat, sat} (3 tokens)
        # reference tokens: {the, cat, sat, on, the, mat} (6 tokens, "the" x2)
        # overlap = min(count in pred, count in ref) summed over pred tokens
        #   the: min(1,2)=1 ; cat: min(1,1)=1 ; sat: min(1,1)=1 -> overlap=3
        # precision = 3/3 = 1.0 ; recall = 3/6 = 0.5
        # f1 = 2*1.0*0.5 / (1.0+0.5) = 1.0/1.5 = 0.6666...
        result = token_overlap_f1("the cat sat", "the cat sat on the mat")
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert abs(result["f1"] - (2 / 3)) < 1e-9

    def test_empty_prediction_scores_zero_not_error(self):
        result = token_overlap_f1("", "some reference text")
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_case_insensitive(self):
        result = token_overlap_f1("The CAT", "the cat")
        assert result["f1"] == 1.0

    def test_is_symmetric_in_definition_but_not_value(self):
        # precision/recall swap when arguments swap; f1 is symmetric.
        forward = token_overlap_f1("the cat sat", "the cat sat on the mat")
        backward = token_overlap_f1("the cat sat on the mat", "the cat sat")
        assert forward["precision"] == backward["recall"]
        assert forward["recall"] == backward["precision"]
        assert abs(forward["f1"] - backward["f1"]) < 1e-9


class TestRougeLF1:
    def test_identical_strings_score_one(self):
        assert rouge_l_f1("nimbus cloud is great", "nimbus cloud is great") == 1.0

    def test_completely_disjoint_scores_zero(self):
        assert rouge_l_f1("apples oranges", "trucks cars") == 0.0

    def test_hand_computed_lcs(self):
        # pred: [a, b, c, d]   ref: [a, x, b, y, c]
        # LCS = [a, b, c] -> length 3
        # recall = 3/5 = 0.6 ; precision = 3/4 = 0.75
        # beta=1.2 -> beta^2=1.44
        # F = (1+1.44)*0.6*0.75 / (0.6 + 1.44*0.75) = 2.44*0.45 / (0.6+1.08)
        #   = 1.098 / 1.68 = 0.653571...
        score = rouge_l_f1("a b c d", "a x b y c", beta=1.2)
        assert abs(score - 0.653571) < 1e-4

    def test_empty_strings_score_zero(self):
        assert rouge_l_f1("", "something") == 0.0
        assert rouge_l_f1("something", "") == 0.0

    def test_reordering_reduces_lcs_based_score_vs_bow(self):
        # ROUGE-L is order-sensitive: reversing word order should not give
        # a perfect score even though the bag of words is identical.
        pred = "cat sat mat"
        ref = "mat sat cat"
        assert rouge_l_f1(pred, ref) < 1.0


class TestKeywordRecall:
    def test_all_facts_found(self):
        result = keyword_recall(
            "The Team plan costs $49 per user per month.", ["$49", "per user"]
        )
        assert result["recall"] == 1.0
        assert result["missed"] == []
        assert set(result["found"]) == {"$49", "per user"}

    def test_some_facts_missing(self):
        result = keyword_recall("The plan costs money.", ["$49", "per user"])
        assert result["recall"] == 0.0
        assert result["found"] == []
        assert set(result["missed"]) == {"$49", "per user"}

    def test_case_insensitive_matching(self):
        result = keyword_recall("We are SOC 2 Type II certified.", ["soc 2 type ii"])
        assert result["recall"] == 1.0

    def test_empty_key_facts_is_trivially_satisfied(self):
        result = keyword_recall("anything at all", [])
        assert result["recall"] == 1.0

    def test_partial_recall_fraction(self):
        result = keyword_recall("only fact one is here", ["fact one", "fact two", "fact three"])
        assert abs(result["recall"] - (1 / 3)) < 1e-9


class TestScoreExample:
    def test_passes_when_keyword_recall_meets_threshold(self):
        score = score_example(
            example_id="ex-1",
            question="How much is the Team plan?",
            prediction="The Team plan is $49 per user per month.",
            reference="The Team plan costs $49 per user per month.",
            key_facts=["$49"],
            pass_threshold=0.5,
        )
        assert score.passed is True
        assert score.keyword_recall == 1.0

    def test_fails_when_keyword_recall_below_threshold(self):
        score = score_example(
            example_id="ex-2",
            question="How much is the Team plan?",
            prediction="I'm not sure about pricing.",
            reference="The Team plan costs $49 per user per month.",
            key_facts=["$49", "per user"],
            pass_threshold=0.5,
        )
        assert score.passed is False
        assert score.keyword_recall == 0.0
        assert set(score.keyword_missed) == {"$49", "per user"}
