"""Unit tests for services (LLMClient, SearchClient, LocalArtifactStore)."""

import tempfile
from pathlib import Path

from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore


def test_llm_client_fallback() -> None:
    client = LLMClient()
    resp = client.complete("You are a writer.", "Summarize multi-agent systems.")
    assert resp.content
    assert resp.input_tokens is not None
    assert resp.output_tokens is not None
    assert resp.cost_usd is not None


def test_search_client_mock() -> None:
    client = SearchClient()
    docs = client.search("Research GraphRAG state-of-the-art", max_results=3)
    assert len(docs) == 3
    assert docs[0].title
    assert docs[0].snippet


def test_local_artifact_store() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LocalArtifactStore(root=Path(tmpdir))
        file_path = store.write_text("test_report.md", "# Test Content")
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == "# Test Content"
