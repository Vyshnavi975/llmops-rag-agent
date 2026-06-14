# LLMOps RAG Agent

An agentic retrieval-augmented-generation system, built on **LangChain** and
**LangGraph**, with the LLMOps tooling a production deployment actually
needs wrapped around it: versioned prompts, an automated evaluation
harness with hand-implemented metrics, per-call token/cost tracking, and
structured run tracing.

The scenario is a support assistant over a fictional SaaS company's
internal knowledge base ("Nimbus Cloud" — see `sample_docs/`), but nothing
about the code is specific to that scenario; point `ingest` at any folder
of `.md`/`.txt` files.

**Runs out of the box with zero setup and zero API key.** With no
`OPENAI_API_KEY` set, the whole system — embeddings, LLM, ingest, ask,
eval, trace — runs fully offline against deterministic, dependency-free
implementations. Set `OPENAI_API_KEY` and the exact same code paths switch
to real `OpenAIEmbeddings` and `ChatOpenAI` with no other changes. Every
command and every number in this README's **Usage** section below was
actually run in demo mode in this repository — nothing is fabricated.

---

## Contents

- [Architecture](#architecture)
- [Why these LangChain/LangGraph choices](#why-these-langchainlanggraph-choices)
- [Setup](#setup)
- [Usage](#usage-real-captured-output)
- [The self-check retry loop, concretely](#the-self-check-retry-loop-concretely)
- [LLMOps design choices](#llmops-design-choices)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations and honest trade-offs](#limitations-and-honest-trade-offs)

---

## Architecture

```
                              START
                                |
                                v
                     +----------------------+
                     |  decide_retrieval    |   heuristic: does this
                     |                      |   question need the KB?
                     +----------------------+
                                |
                needs_retrieval?|
              +-----------------+------------------+
              | yes                                 | no
              v                                      v
      +---------------+                     +----------------+
      |   retrieve    | <-------- retry ---- |    generate    |
      +---------------+                      +----------------+
              |                                      ^
              v                                      |
      +---------------+                              |
      |    generate   | -----------------------------+
      +---------------+
              |
              v
      +---------------+
      |  self_check   |
      +---------------+
        |                              |
        | passed, or retries exhausted | failed AND retries remain
        v                              v
       END                    back to `retrieve`
                        (broadened query, larger k)
```

This is a real `langgraph.graph.StateGraph` (`agent/graph.py`) with four
nodes and two conditional edges — not a linear pipeline wearing a graph
costume:

| Node | Responsibility |
|---|---|
| `decide_retrieval` | Lightweight heuristic router: skip the vector store entirely for greetings/meta questions ("hi", "what can you do?") instead of paying a retrieval + generation cost for them. |
| `retrieve` | Calls the RAG pipeline's vector store (`rag/store.py`) for the current query and `k`. Runs 1–2 times per request depending on `self_check`. |
| `generate` | Builds a prompt from the versioned prompt registry (`ops/prompt_registry.py`) and calls the LLM (`ChatOpenAI` or `FakeListChatModel`) via a real LCEL-style call. |
| `self_check` | A grounding check: are the question's key terms actually present in the retrieved context, and did the model avoid an explicit "I don't know"? If not, and a retry budget remains, it rewrites the query (appends the missing terms) and widens `k`, and the graph loops back to `retrieve`. |

`self_check → retrieve → generate → self_check → END` is a genuine
conditional loop compiled by LangGraph's `add_conditional_edges`, bounded
by `--max-retries` (default 1) so it can't loop forever. See
[The self-check retry loop, concretely](#the-self-check-retry-loop-concretely)
for an actual run where this fires.

### The three modules

```
rag/    document ingestion, chunking, embeddings, vector store
agent/  the LangGraph StateGraph described above
ops/    prompt registry, eval harness, cost tracking, tracing  (the "LLMOps" layer)
```

`ops/` doesn't know anything about RAG or LangGraph specifically — it's
generic run-accounting infrastructure (prompt versions, cost, traces, eval
scoring) that `agent/graph.py` calls into. In a larger system it's the
layer that would be shared across multiple agents/pipelines.

## Why these LangChain/LangGraph choices

**Vector store: `langchain_core.vectorstores.InMemoryVectorStore`, not
FAISS.** Both were actually evaluated in this environment. `faiss-cpu`
installs cleanly, but the LangChain FAISS integration currently lives only
in `langchain-community`, which prints a deprecation warning on import
("being sunset ... in favor of standalone integration packages"). Building
a new portfolio project on an integration the maintainers are actively
moving away from is the wrong trade-off when `InMemoryVectorStore` — a
first-class citizen of `langchain-core` itself, with the identical
`VectorStore` interface (`add_documents`, `similarity_search_with_score`,
`as_retriever`, ...) and built-in JSON persistence via `.dump()`/`.load()`
— is right there with no extra dependency and no warning. For a
knowledge base of a few dozen chunks its brute-force cosine search is not
a real accuracy or latency compromise. Swapping in FAISS or a hosted
vector DB later touches exactly two files (`rag/ingest.py`, `rag/store.py`)
because everything else in the agent graph only depends on LangChain's
`VectorStore`/`Embeddings` interfaces, not on the concrete class.

**Embeddings: real `OpenAIEmbeddings`, or a from-scratch `Embeddings`
implementation.** `rag/embeddings.py`'s `HashingEmbeddings` implements
LangChain's `Embeddings` ABC using the hashing trick (tokens hashed into a
fixed-size vector with a sign derived from a second hash, plus character
n-gram features, L2-normalized) — fully deterministic, no model download,
no network call. It is not a neural embedding and its retrieval quality is
noticeably weaker than a real model (see
[Limitations](#limitations-and-honest-trade-offs) — this is the direct
cause of two eval failures and of the retry-loop example below), but it
lets `ingest`/`ask`/`eval`/`trace` all genuinely run end-to-end with zero
setup, which is the whole point of a demo mode.

**LLM: real `ChatOpenAI`, or LangChain's `FakeListChatModel`.**
`FakeListChatModel` is a real `BaseChatModel` subclass from
`langchain_core`, not a hand-rolled stand-in — it drops into the exact same
graph node, called the exact same way (`llm.invoke(prompt_text)`), as
`ChatOpenAI` does. In demo mode, `agent/demo_answers.py` computes a
deterministic, context-dependent extractive answer *before* each call and
pushes it onto the fake model's response queue (`set_fake_response`), so
what's "fake" is the model's reasoning — the graph's control flow
(routing, retrieval, prompt construction, the self-check loop and its
retries) all execute for real. This is also what makes the eval numbers
below fully reproducible: rerun `eval` ten times in demo mode and every
score is bit-for-bit identical, because every component in the pipeline is
deterministic.

**Why prompt version literally changes the demo model's behavior.**
`answer_v1.txt` asks for a plain grounded answer; `answer_v2.txt`
additionally asks for 2–4 concise sentences covering concrete facts, with a
`[source.md]` citation. In demo mode, `agent/demo_answers.py` reads that
same distinction directly (`cite_sources = resolved_version >= 2`) and
selects 1 sentence for v1 vs. up to 3 + a citation for v2 — so switching
prompt versions produces a genuinely different, measurably different
answer and eval score, standing in for what a real prompt edit would do to
a real LLM's output. See the v1-vs-v2 eval comparison in
[Usage](#usage-real-captured-output).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the test suite

# Optional — omit this to run entirely offline in demo mode:
export OPENAI_API_KEY=sk-...
```

Python 3.11 was used to build and test this project; anything 3.10+ should
work given the pinned dependency versions in `requirements.txt`
(`langchain==1.4.0`, `langgraph==1.2.11`, `langchain-openai==1.6.0`).

## Usage (real captured output)

Every block below is unedited output from actually running these commands
against this repository's `sample_docs/`, with no `OPENAI_API_KEY` set.

### 1. `ingest` — build the vector index

```
$ python3 cli.py ingest --docs sample_docs --index-dir data/index
Ingested 6 document(s) into 21 chunk(s).
Embedding backend: offline:hashing-bow
Index written to:  ./data/index
Elapsed:            0.021s
```

### 2. `ask` — run the full agent graph on one question

```
$ python3 cli.py ask --question "How much does the Team plan cost per user per month?"
Mode:            demo (offline)   Model: demo-fake-llm
Prompt:          answer_v2
Needs retrieval: True
Retrieval passes: 1   Self-check passed: True
Sources:         api_reference.md, pricing_and_billing.md

Answer:
Customers who commit to annual billing instead of monthly billing receive a 15% discount on the Team and Enterprise plans. Storage usage beyond a plan's included quota is billed at $0.08 per GB per month on the Team plan. Team — $49 per user per month. [pricing_and_billing.md]

Tokens: 536 (in=475, out=61)   Cost: $0.000000
Trace saved to:  traces/20260903T215318-f0d4382e.json
```

A question that doesn't need the knowledge base at all skips retrieval
entirely (`decide_retrieval`'s conditional edge routes straight to
`generate`):

```
$ python3 cli.py ask --question "Hi there, what can you help me with?"
Mode:            demo (offline)   Model: demo-fake-llm
Prompt:          answer_v2
Needs retrieval: False
Retrieval passes: 0   Self-check passed: True

Answer:
I can answer questions about Nimbus Cloud's products, pricing, security and compliance, onboarding and support, API, and incident response processes. Please ask a specific question and I'll look it up.

Tokens: 187 (in=148, out=39)   Cost: $0.000000
Trace saved to:  traces/20260903T215323-2610442d.json
```

### 3. `eval` — score the agent against the labeled eval set

```
$ python3 cli.py eval
Prompt: answer_v2   Model: demo-fake-llm (demo mode)
Pass rate: 86.7%   Avg F1: 0.337   Avg ROUGE-L: 0.343   Avg keyword recall: 88.9%
Retries triggered: 1/15   Total tokens: 8540   Total cost: $0.000000   Elapsed: 0.29s

id         pass       f1  rouge_l  kw_recall  question
----------------------------------------------------------------------------------------------------
eval-01    PASS    0.267    0.305     100.0%  How much does the Team plan cost per user per month?
eval-02    PASS    0.351    0.394     100.0%  What discount do I get for paying annually instead of monthl
eval-03    PASS    0.256    0.295     100.0%  Is Nimbus Cloud SOC 2 certified?
eval-04    PASS    0.340    0.192     100.0%  What encryption does Nimbus Cloud use for data at rest and i
eval-05    PASS    0.277    0.292     100.0%  What is the response SLA for a severity-1 incident under the
eval-06    PASS    0.415    0.419     100.0%  How many API requests per minute can a Team plan API key mak
eval-07    PASS    0.271    0.312     100.0%  How long does Nimbus Cloud retain customer data after an acc
eval-08    FAIL    0.136    0.155       0.0%  What is Nimbus Cloud's uptime target for its core API?
                  missed key facts: ['99.9%']
eval-09    FAIL    0.252    0.188      33.3%  Within how many business days is a SEV1 postmortem published
                  missed key facts: ['BAA', 'HIPAA']
eval-10    PASS    0.238    0.247     100.0%  What two AWS-style regions does Nimbus Cloud operate in?
eval-11    PASS    0.424    0.447     100.0%  Do I need a credit card to start a free Starter-tier account
eval-12    PASS    0.417    0.435     100.0%  What happens if a webhook endpoint fails to respond in time?
eval-13    PASS    0.366    0.382     100.0%  What four built-in RBAC roles does Nimbus Cloud support?
eval-14    PASS    0.250    0.256     100.0%  What is Nimbus Vault used for and how often does it rotate s
eval-15    PASS    0.792    0.823     100.0%  Hi there, what can you help me with?

Full JSON report saved to: reports/eval_answer_v2_20260903T215330.json
```

**Comparing prompt versions on the same eval set** (`--prompt-version 1`)
is exactly what this harness is for:

```
$ python3 cli.py eval --prompt-version 1 --no-traces
Prompt: answer_v1   Model: demo-fake-llm (demo mode)
Pass rate: 80.0%   Avg F1: 0.608   Avg ROUGE-L: 0.605   Avg keyword recall: 81.1%
Retries triggered: 1/15   Total tokens: 6589   Total cost: $0.000000   Elapsed: 0.25s
```

|  | `answer_v1` | `answer_v2` | Delta |
|---|---:|---:|---:|
| Pass rate | 80.0% | **86.7%** | +6.7 pts |
| Avg keyword recall | 81.1% | **88.9%** | +7.8 pts |
| Avg F1 (vs. reference) | **0.608** | 0.337 | -0.271 |
| Avg ROUGE-L | **0.605** | 0.343 | -0.262 |
| Total tokens (15 examples) | 6,589 | 8,540 | +30% |

This is a real, explainable trade-off, not a coincidence of the demo
heuristic: `v1` extracts one sentence, so when it's the *right* sentence it
matches a one-sentence reference answer very closely (high F1/ROUGE-L); `v2`
extracts up to three sentences plus a citation, so it covers more of the
specific facts a real user question tends to need (higher keyword recall,
higher pass rate — `eval-01` needs both "$49" *and* "per user", which `v1`
misses and `v2` catches) at the cost of similarity to a single reference
phrasing, and at a real token/cost increase. That is precisely the kind of
decision — "do we want higher fact coverage or tighter, cheaper answers?"
— an eval harness is supposed to make visible instead of hidden behind
one engineer's subjective read of a handful of example outputs.

### 4. `trace` — inspect the most recent run in detail

```
$ python3 cli.py trace --run-id 20260903T215321-936b9936
Run ID:        20260903T215321-936b9936
  question: What two AWS-style regions does Nimbus Cloud operate in?
  prompt: answer_v2
  model: demo-fake-llm
  mode: demo
  embedding_backend: offline:hashing-bow
Wall clock:    256.68 ms
Node events:   7

#   node             latency_ms  input -> output
----------------------------------------------------------------------------------------------------
1   decide_retrieval       0.08  What two AWS-style regions does Nimbus Cloud operate in?
    -> needs_retrieval=True (informational question -> retrieval required)
2   retrieve               1.04  query='What two AWS-style regions does Nimbus Cloud operate in?', k=3
    -> 3 chunk(s) from ['pricing_and_billing.md', 'product_overview.md', 'security_and_compliance.md']
    [{'retrieval_pass': 1, 'k': 3}]
3   generate             247.41  prompt=answer_v2, context_chars=1157
    -> Security is treated as a product feature at Nimbus Cloud, not an afterthought. Nimbus Cloud has completed a SOC 2 Type II audit and can provide the report to customers under NDA. Nimbus Cloud is also GDPR compliant fo...
    [{'model': 'demo-fake-llm'}]
4   self_check             0.05  answer='Security is treated as a product feature at Nimbus Cloud, not an afterthought. N'
    -> passed=False (grounding_score=0.40 over 5 question keyword(s)); retry=True
    [{'grounding_score': 0.4, 'missing_keywords': ['style', 'regions', 'operate']}]
5   retrieve               1.26  query='What two AWS-style regions does Nimbus Cloud operate in? style regions operate', k=6
    -> 6 chunk(s) from ['pricing_and_billing.md', 'product_overview.md', 'security_and_compliance.md']
    [{'retrieval_pass': 2, 'k': 6}]
6   generate               0.81  prompt=answer_v2, context_chars=3253
    -> Nimbus Cloud runs on top of two primary regions today: `us-east-1` and `eu-west-1`, with additional regions on the public roadmap. Nimbus Cloud also operates a public bug-bounty program for responsible disclosure of s...
    [{'model': 'demo-fake-llm'}]
7   self_check             0.03  answer='Nimbus Cloud runs on top of two primary regions today: `us-east-1` and `eu-west-'
    -> passed=True (grounding_score=0.80 over 5 question keyword(s)); retry=False
    [{'grounding_score': 0.8, 'missing_keywords': ['style']}]

Final answer: Nimbus Cloud runs on top of two primary regions today: `us-east-1` and `eu-west-1`, with additional regions on the public roadmap. Nimbus Cloud also operates a public bug-bounty program for responsible disclosure of security issues. Nimbus Cloud is aimed at small-to-midsize engineering teams (typically 5 to 500 engineers) that want production-grade data infrastructure without hiring a dedicated...

Cost:          2 LLM call(s), 1235 tokens, $0.0
```

## The self-check retry loop, concretely

The trace above is `eval-10` ("What two AWS-style regions does Nimbus Cloud
operate in?") from the eval run, and it's a genuine example of the
conditional loop firing at runtime — not a scripted demonstration:

1. **`retrieve` (pass 1, k=3)** pulls 3 chunks. With the offline hashing
   embedding, the query's dominant tokens ("AWS-style", "regions",
   "operate") don't land it near the one chunk in `product_overview.md`
   that actually names the two regions — instead it surfaces plausible
   but off-target chunks from `security_and_compliance.md` and
   `pricing_and_billing.md`.
2. **`generate`** produces an answer grounded in *that* context — reasonable
   given what it was handed, but it doesn't answer the question.
3. **`self_check`** computes a grounding score (fraction of the question's
   keywords — `style`, `regions`, `operate`, `aws`, `nimbus` — that
   actually appear in the retrieved context): **0.40**, below the 0.5 pass
   threshold. `self_check_passed = False`, and a retry budget remains
   (`max_retries=1`, this is the first attempt), so the conditional edge
   routes back to `retrieve` instead of `END`.
4. **`retrieve` (pass 2, k=6)** runs again with a rewritten, broadened
   query (the missing keywords appended) and a larger `k` — and this time
   surfaces the right chunk, from `product_overview.md`, containing
   "`us-east-1`" and "`eu-west-1`".
5. **`generate`** now produces a correctly grounded answer.
6. **`self_check`** recomputes grounding at **0.80** — passes — and the
   conditional edge routes to `END`.

Net effect: one extra retrieval + generation pass, two extra LLM calls'
worth of token cost, and a wrong-context answer silently repaired into a
correct one — the entire reason to build a "self-check → retry" node
instead of a fixed retrieve-once pipeline. `tests/test_agent_graph.py::TestSelfCheckRetryLoop::test_self_check_triggers_a_real_retry_and_recovers`
asserts on this exact run deterministically (both hashing embeddings and
the demo heuristics are 100% reproducible, so this behavior is not
flaky — it happens on every run of `eval`, every time).

## LLMOps design choices

**Prompt registry (`ops/prompt_registry.py`, `prompts/*.txt`).** Prompts
live in their own version-numbered files instead of Python string literals
buried in `agent/graph.py`. That gives a prompt edit the same properties as
a code change: it's a diffable, reviewable line in `git log`; an eval run,
a trace, or an incident postmortem can cite exactly `answer_v2` instead of
"whatever the prompt happened to say at 3pm on Tuesday"; and two versions
can be evaluated side by side (see the v1-vs-v2 table above) without
checking out a different commit. In a team setting this is also what makes
a prompt change reviewable by someone who isn't the person who wrote it.

**Evaluation harness (`ops/eval_harness.py`, `ops/metrics.py`,
`ops/eval_set.json`).** A prompt or model change that "feels better" on
three manually-typed test questions is not evidence — it's an anecdote.
The harness runs a fixed, labeled 15-question set through the *actual*
compiled agent graph on every invocation and reports a pass rate plus three
independent similarity signals (token-overlap F1, ROUGE-L, and
fact-specific keyword recall — deliberately three, since they disagree in
informative ways, as the v1-vs-v2 table shows), so a prompt/model change is
judged by a before/after number instead of a feeling. All three metrics are
implemented from scratch in `ops/metrics.py` (no `evaluate`/`rouge-score`
dependency) specifically so the eval harness has zero external
dependencies and zero network calls — it has to be runnable in CI on every
pull request that touches a prompt, with no API key available.

**Cost/token tracking (`ops/cost.py`).** Every `generate` call is metered:
tokens counted with `tiktoken` when available (falling back to a
documented word-based approximation — and falling back automatically at
runtime, not just at install time, if `tiktoken`'s BPE files can't be
fetched, which matters in an air-gapped or network-restricted environment
exactly like the one this project was built in) and priced against an
explicit per-model table. This is the number that turns "let's just always
retrieve `k=10` and add a second self-check pass" from a free-sounding
idea into a line item — in a real deployment, the difference between a
$0.0002 and a $0.02 answer, multiplied by call volume, is the difference
between a sustainable feature and a surprise on next month's OpenAI bill.
Aggregating it per-run (and exposing it in both `ask`'s output and the
`eval` report's `total_cost_usd`) is what lets a model swap or a retry-limit
change be evaluated on cost *and* quality together, which is the trade-off
that actually matters in production.

**Structured tracing (`ops/tracing.py`, `traces/*.json`).** Every run
writes one JSON file recording each graph node entered (including repeat
entries from a retry — a fixed pipeline can't produce that, which is
exactly why it's worth recording), a truncated input/output summary,
per-node latency, and the run's aggregate token/cost. This is the artifact
an engineer reaches for when a user reports "the bot gave a weird answer":
instead of trying to reproduce the exact retrieval that happened at 2am,
`cli.py trace --run-id <id>` replays the graph's decisions from the record.
It's also what made the concrete walkthrough in the previous section
possible to write truthfully instead of describing hypothetical behavior.

## Project layout

```
llmops-rag-agent/
├── agent/
│   ├── graph.py           # the LangGraph StateGraph (nodes, conditional edges)
│   ├── state.py            # AgentState TypedDict
│   ├── llm.py               # ChatOpenAI / FakeListChatModel selection
│   ├── demo_answers.py       # deterministic extractive answers for demo mode
│   └── heuristics.py          # keyword extraction, greeting detection, markdown cleanup
├── rag/
│   ├── embeddings.py        # OpenAIEmbeddings / HashingEmbeddings
│   ├── ingest.py             # load -> chunk (RecursiveCharacterTextSplitter) -> embed -> persist
│   └── store.py               # load a persisted index, similarity search, retriever
├── ops/
│   ├── prompt_registry.py    # versioned prompt loading
│   ├── metrics.py             # token-overlap F1, ROUGE-L, keyword recall (from scratch)
│   ├── eval_harness.py         # run the eval set through the agent, score, report
│   ├── cost.py                  # token counting + cost estimation
│   ├── tracing.py                # structured JSON run traces
│   └── eval_set.json              # 15 labeled Q/A examples
├── prompts/
│   ├── answer_v1.txt
│   └── answer_v2.txt
├── sample_docs/               # 6 short fictional-company knowledge-base docs
├── tests/                     # 63 tests, all offline / no API key
├── cli.py                     # ingest / ask / eval / trace subcommands
├── data/index/                 # generated by `ingest` (gitignored)
├── traces/                     # generated per run (gitignored)
├── reports/                    # generated per eval run (gitignored)
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Testing

```
$ python3 -m pytest -v
...
============================== 63 passed in 0.77s ==============================
```

63 tests across four files, none requiring an API key or network access:

- `tests/test_metrics.py` — hand-computed expected values for
  token-overlap F1, ROUGE-L (including a worked LCS-by-hand example), and
  keyword recall.
- `tests/test_cost.py` — hand-computed token counts for the word-based
  fallback approximation (with `tiktoken` monkeypatched off so the test is
  deterministic regardless of whether it's installed), price-table math,
  and `CostTracker` aggregation.
- `tests/test_prompt_registry.py` — version discovery and numeric (not
  lexicographic — `v10` must beat `v2`) "latest" resolution, exact-content
  loading, and error behavior for missing prompts/versions.
- `tests/test_rag_pipeline.py` — `HashingEmbeddings` determinism and
  normalization, document loading/chunking, and index build/persist/reload
  round-trips with real similarity search assertions.
- `tests/test_agent_graph.py` — end-to-end runs of the real compiled
  `StateGraph`: routing skips retrieval for greetings, an informational
  question triggers retrieval, the self-check retry loop fires and
  recovers on the documented example (asserting on the actual trace event
  sequence), retries are bounded by `max_retries`, and prompt version
  changes the model's cited-sources behavior.

## Limitations and honest trade-offs

- **The offline hashing embedding is intentionally weak.** It has no
  notion of synonymy or semantics beyond shared tokens/character n-grams,
  which is why `eval-08` and `eval-09` fail in demo mode (the retrieved
  context genuinely doesn't contain "99.9%" or "HIPAA"/"BAA" together with
  the other retrieved facts) and why `eval-10` needs a retry to recover.
  This is not hidden or scored around — the eval report above is the real,
  unedited number. With `OPENAI_API_KEY` set (real `OpenAIEmbeddings`),
  retrieval quality — and therefore pass rate — would be expected to
  improve; that comparison is left to whoever runs this with a real key,
  since fabricating those numbers here would defeat the point of this
  README's "everything is real, captured output" claim.
- **The self-check node is a heuristic (keyword grounding), not an LLM
  call**, in both demo and live mode. This is a deliberate cost/latency
  choice, not a limitation of demo mode specifically: a cheap, deterministic
  grounding check catches the most common agentic-RAG failure (context
  doesn't cover the question) without spending a second paid LLM call on
  every single request just to ask "did you answer the question?". A real
  deployment with a larger self-check budget could swap this node for an
  LLM-as-judge call with no change to the graph's edges.
- **Demo mode's "reasoning" is extractive, not generative** — it selects
  and lightly cleans real sentences from the retrieved context rather than
  synthesizing new phrasing. That's by design (see
  [Why these LangChain/LangGraph choices](#why-these-langchainlanggraph-choices)):
  it keeps every answer verifiably grounded and fully deterministic for
  reproducible eval numbers, at the cost of the fluency a real LLM would
  add.

## License

MIT — see [LICENSE](LICENSE).
