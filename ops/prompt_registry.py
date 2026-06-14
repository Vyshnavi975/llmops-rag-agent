"""Prompt registry: treat prompts as versioned artifacts, not string literals
buried in application code.

Every prompt lives in its own file under `prompts/`, named
`{name}_v{version}.txt`. That gives prompt changes the same properties as
code changes: they are diffable in `git log`, reviewable in a pull request,
and — critically for LLMOps — addressable by an exact version, so an eval
run, a trace, or an incident report can say precisely "this response was
generated with prompt `answer` version `2`" instead of "whatever the prompt
happened to say at the time." Bumping a prompt to a new file (rather than
editing in place) also means an eval harness can compare two versions
side-by-side without checking out a different commit.

`PromptRegistry` is a thin loader around that convention: it discovers
available versions by globbing the prompts directory, resolves "latest" to
the highest version number, and wraps the loaded text in a LangChain
`PromptTemplate` so it can be dropped straight into an LCEL chain.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from langchain_core.prompts import PromptTemplate

_FILENAME_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_\-]+)_v(?P<version>\d+)\.txt$")


class PromptNotFoundError(RuntimeError):
    pass


class PromptRegistry:
    def __init__(self, prompts_dir: str = "prompts") -> None:
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    def _index(self) -> Dict[str, Dict[int, Path]]:
        """Map prompt name -> {version_int: path}, discovered fresh each
        call so newly-added prompt files are picked up without restarting
        anything (prompts are meant to be edited like code)."""
        index: Dict[str, Dict[int, Path]] = {}
        for path in self.prompts_dir.glob("*.txt"):
            match = _FILENAME_RE.match(path.name)
            if not match:
                continue
            name = match.group("name")
            version = int(match.group("version"))
            index.setdefault(name, {})[version] = path
        return index

    def list_names(self) -> List[str]:
        return sorted(self._index().keys())

    def list_versions(self, name: str) -> List[int]:
        versions = self._index().get(name, {})
        if not versions:
            raise PromptNotFoundError(f"No prompt found with name '{name}'")
        return sorted(versions.keys())

    def resolve_version(self, name: str, version: str | int = "latest") -> int:
        versions = self.list_versions(name)
        if version == "latest":
            return max(versions)
        version_int = int(version)
        if version_int not in versions:
            raise PromptNotFoundError(
                f"Prompt '{name}' has no version {version_int}. Available: {versions}"
            )
        return version_int

    def get_text(self, name: str, version: str | int = "latest") -> str:
        resolved = self.resolve_version(name, version)
        path = self._index()[name][resolved]
        return path.read_text(encoding="utf-8")

    def get_template(self, name: str, version: str | int = "latest") -> PromptTemplate:
        """Return the prompt as a LangChain `PromptTemplate`, with input
        variables inferred from `{placeholders}` in the file."""
        text = self.get_text(name, version)
        return PromptTemplate.from_template(text)

    def get_version_label(self, name: str, version: str | int = "latest") -> str:
        resolved = self.resolve_version(name, version)
        return f"{name}_v{resolved}"
