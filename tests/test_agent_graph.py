"""End-to-end tests for the LangGraph agent graph, in demo mode (no API key,
no network). These exercise the *real* compiled `StateGraph` — routing,
retrieval, generation via `FakeListChatModel`, and the self-check
conditional loop — not a mocked-out substitute.
"""

from pathlib import Path

import pytest

from agent.graph import build_agent_graph, run_agent
from rag.ingest import build_index

SAMPLE_DOCS_DIR = str(Path(__file__).resolve().parent.parent / "sample_docs")


@pytest.fixture(autouse=True)
def no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("index"))
    build_index(docs_dir=SAMPLE_DOCS_DIR, index_dir=d)
    return d


@pytest.fixture(scope="module")
def bundle(index_dir):
    return build_agent_graph(index_dir=index_dir, prompt_name="answer", prompt_version="latest")


@pytest.fixture
def traces_dir(tmp_path):
    return str(tmp_path / "traces")


class TestDemoModeSelection:
    def test_bundle_is_in_demo_mode_without_api_key(self, bundle):
        assert bundle.is_demo is True
        assert bundle.model_name == "demo-fake-llm"


class TestRoutingSkipsRetrievalForGreetings(object):
    def test_greeting_skips_retrieval(self, bundle, traces_dir):
        result = run_agent(bundle, "Hi there, what can you help me with?", traces_dir=traces_dir)
        assert result.needs_retrieval is False
        assert result.retrieval_count == 0
        assert result.sources == []
        assert "Nimbus Cloud" in result.answer

    def test_informational_question_triggers_retrieval(self, bundle, traces_dir):
        result = run_agent(bundle, "How much does the Team plan cost per user per month?", traces_dir=traces_dir)
        assert result.needs_retrieval is True
        assert result.retrieval_count >= 1
        assert "pricing_and_billing.md" in result.sources


class TestSelfCheckRetryLoop:
    def test_self_check_triggers_a_real_retry_and_recovers(self, bundle, traces_dir):
        """This question is deliberately hard for a single k=3 retrieval
        pass against the offline hashing embeddings: the first pass
        surfaces plausible-looking but off-target chunks, the grounding
        self-check fails, and the graph loops back into `retrieve` with a
        broadened query and larger k. This is a real conditional edge
        firing at runtime, not a scripted assertion — the fixture asserts
        on the *outcome* of that loop, deterministically, since every
        component involved (hashing embeddings, heuristics, fake LLM) is
        deterministic.
        """
        question = "What two AWS-style regions does Nimbus Cloud operate in?"
        result = run_agent(bundle, question, traces_dir=traces_dir)

        assert result.retrieval_count == 2, "expected exactly one self-check-triggered retry"
        assert result.self_check_passed is True, "the retry should have recovered a grounded answer"
        assert "us-east-1" in result.answer
        assert "eu-west-1" in result.answer

        # The trace should literally show two 'retrieve' node events and a
        # 'self_check' event in between reporting the failed first pass.
        retrieve_events = [e for e in result.trace["events"] if e["node"] == "retrieve"]
        self_check_events = [e for e in result.trace["events"] if e["node"] == "self_check"]
        assert len(retrieve_events) == 2
        assert len(self_check_events) == 2
        assert retrieve_events[0]["k"] < retrieve_events[1]["k"], "retry should widen retrieval (larger k)"
        assert "passed=False" in self_check_events[0]["output_summary"]
        assert "passed=True" in self_check_events[1]["output_summary"]

    def test_bounded_by_max_retries(self, index_dir, traces_dir):
        """With max_retries=0, self-check must never trigger a second
        retrieval pass, even for a question that would otherwise retry."""
        bundle_no_retry = build_agent_graph(index_dir=index_dir, max_retries=0)
        question = "What two AWS-style regions does Nimbus Cloud operate in?"
        result = run_agent(bundle_no_retry, question, traces_dir=traces_dir)
        assert result.retrieval_count == 1


class TestCostAndTracing:
    def test_run_produces_a_trace_file_and_cost_summary(self, bundle, traces_dir):
        result = run_agent(bundle, "Is Nimbus Cloud SOC 2 certified?", traces_dir=traces_dir)
        assert result.trace_path is not None
        assert Path(result.trace_path).exists()
        assert result.cost_summary["num_llm_calls"] >= 1
        # Demo mode is always free.
        assert result.cost_summary["total_cost_usd"] == 0.0

    def test_prompt_version_changes_citation_behavior(self, index_dir, traces_dir):
        bundle_v1 = build_agent_graph(index_dir=index_dir, prompt_version=1)
        bundle_v2 = build_agent_graph(index_dir=index_dir, prompt_version=2)
        question = "Is Nimbus Cloud SOC 2 certified?"

        result_v1 = run_agent(bundle_v1, question, traces_dir=traces_dir)
        result_v2 = run_agent(bundle_v2, question, traces_dir=traces_dir)

        assert bundle_v1.prompt_label == "answer_v1"
        assert bundle_v2.prompt_label == "answer_v2"
        # v2's prompt instructs the (demo) generator to cite sources in brackets; v1's does not.
        assert "[" not in result_v1.answer
        assert "[security_and_compliance.md]" in result_v2.answer
