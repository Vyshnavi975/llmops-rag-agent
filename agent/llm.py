"""LLM selection: a real `ChatOpenAI` model when `OPENAI_API_KEY` is set, or
LangChain's `FakeListChatModel` for a fully offline demo mode.

The fake model is a genuine `BaseChatModel` subclass from `langchain_core`
(not a hand-rolled stand-in), so it drops into the exact same LCEL chains
(`prompt | llm | StrOutputParser()`) and the exact same LangGraph nodes as
`ChatOpenAI` does — the only difference is where the text it returns comes
from. In demo mode, `agent/demo_answers.py` computes a deterministic,
context-aware response *before* each call and pushes it onto the fake
model's response queue, so the graph's control flow (routing, retrieval,
generation, self-check, retries) is exercised for real, without needing
network access or an API key.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

DEMO_MODEL_NAME = "demo-fake-llm"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def is_demo_mode() -> bool:
    return not bool(os.environ.get("OPENAI_API_KEY"))


def get_llm(model: Optional[str] = None) -> Tuple[BaseChatModel, str, bool]:
    """Return `(llm, model_name, is_demo)`.

    If `OPENAI_API_KEY` is set, returns a real `ChatOpenAI` instance.
    Otherwise returns a `FakeListChatModel` seeded with a placeholder
    response — callers in demo mode should set `llm.responses = [text]`
    (see `set_fake_response` below) immediately before each `.invoke(...)`.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model_name = model or DEFAULT_OPENAI_MODEL
        return ChatOpenAI(model=model_name, temperature=0), model_name, False

    llm = FakeListChatModel(responses=["(demo mode placeholder)"])
    return llm, DEMO_MODEL_NAME, True


def set_fake_response(llm: BaseChatModel, text: str) -> None:
    """Push a single deterministic response onto a `FakeListChatModel`'s
    queue so the next `.invoke(...)` call returns exactly `text`."""
    if not isinstance(llm, FakeListChatModel):
        raise TypeError("set_fake_response only applies to FakeListChatModel instances")
    llm.responses = [text]
    llm.i = 0
