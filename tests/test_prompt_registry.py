"""Unit tests for ops/prompt_registry.py: versioning and loading behavior.

Uses a temporary prompts directory (not the real `prompts/`) so the test
suite doesn't depend on exactly which prompt files happen to exist in the
repository at test time.
"""

import pytest

from ops.prompt_registry import PromptNotFoundError, PromptRegistry


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "answer_v1.txt").write_text("Answer v1. Context: {context} Question: {question}")
    (d / "answer_v2.txt").write_text("Answer v2, improved. Context: {context} Question: {question}")
    (d / "answer_v10.txt").write_text("Answer v10. Context: {context} Question: {question}")
    (d / "router_v1.txt").write_text("Router v1: {question}")
    # A file that doesn't match the naming convention should be ignored.
    (d / "notes.txt").write_text("not a versioned prompt")
    return str(d)


class TestListing:
    def test_list_names_finds_all_prompt_families(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.list_names() == ["answer", "router"]

    def test_list_versions_returns_sorted_ints(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.list_versions("answer") == [1, 2, 10]

    def test_list_versions_unknown_prompt_raises(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        with pytest.raises(PromptNotFoundError):
            registry.list_versions("does_not_exist")


class TestVersionResolution:
    def test_latest_resolves_to_highest_numeric_version(self, prompts_dir):
        # Numeric resolution matters: v10 must beat v2, not sort as "v2" > "v10" lexicographically.
        registry = PromptRegistry(prompts_dir)
        assert registry.resolve_version("answer", "latest") == 10

    def test_explicit_version_resolves_exactly(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.resolve_version("answer", 1) == 1
        assert registry.resolve_version("answer", "2") == 2

    def test_missing_version_raises(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        with pytest.raises(PromptNotFoundError):
            registry.resolve_version("answer", 999)


class TestLoading:
    def test_get_text_returns_exact_file_contents(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.get_text("answer", 1) == "Answer v1. Context: {context} Question: {question}"

    def test_get_text_default_is_latest(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.get_text("answer") == registry.get_text("answer", 10)

    def test_different_versions_have_different_content(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.get_text("answer", 1) != registry.get_text("answer", 2)

    def test_get_template_formats_with_variables(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        template = registry.get_template("answer", 1)
        rendered = template.format(context="CTX", question="Q?")
        assert rendered == "Answer v1. Context: CTX Question: Q?"

    def test_get_version_label(self, prompts_dir):
        registry = PromptRegistry(prompts_dir)
        assert registry.get_version_label("answer", 2) == "answer_v2"
        assert registry.get_version_label("answer", "latest") == "answer_v10"

    def test_missing_prompts_dir_raises_at_construction(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptRegistry(str(tmp_path / "does_not_exist"))

    def test_newly_added_file_is_picked_up_without_reconstruction(self, prompts_dir):
        # The registry re-scans the directory on every call rather than
        # caching an index at construction time, since prompts are meant to
        # be editable like code without restarting anything.
        registry = PromptRegistry(prompts_dir)
        assert registry.resolve_version("answer", "latest") == 10
        from pathlib import Path

        Path(prompts_dir, "answer_v11.txt").write_text("Answer v11. {context} {question}")
        assert registry.resolve_version("answer", "latest") == 11
