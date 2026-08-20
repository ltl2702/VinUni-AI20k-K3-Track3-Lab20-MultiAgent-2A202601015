"""Unit tests for evaluation and benchmark suite."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    estimate_quality_score,
    run_benchmark,
    run_single_agent,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_compute_citation_coverage() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [
        SourceDocument(title="Source 1", snippet="Snippet 1", url="https://example.com/1"),
        SourceDocument(title="Source 2", snippet="Snippet 2", url="https://example.com/2"),
    ]
    # Answer citing [1]
    state.final_answer = "This report discusses findings according to [1] and Source 1."
    cov = compute_citation_coverage(state)
    assert cov == 0.5

    # Answer citing both
    state.final_answer = "This report discusses [1] and [2] in detail."
    cov = compute_citation_coverage(state)
    assert cov == 1.0


def test_estimate_quality_score() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Source 1", snippet="Snippet 1")]
    state.final_answer = (
        "### Overview\n\n"
        "Here is a detailed report with enough depth and content.\n\n"
        "- Point 1: analysis\n"
        "- Point 2: evaluation according to [1]\n\n"
        "### Conclusion\nFinal takeaway."
    )
    score = estimate_quality_score(state)
    assert 0.0 <= score <= 10.0
    assert score >= 5.0


def test_run_single_agent_benchmark() -> None:
    state, metrics = run_benchmark("single_agent", "Test query on LLM guardrails", run_single_agent)
    assert state.final_answer is not None
    assert metrics.run_name == "single_agent"
    assert metrics.latency_seconds >= 0.0

    report = render_markdown_report([metrics])
    assert "single_agent" in report
    assert "Benchmark Report" in report
