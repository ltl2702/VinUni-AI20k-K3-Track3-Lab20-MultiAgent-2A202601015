"""Unit tests for LangGraph multi-agent workflow."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_multi_agent_workflow_end_to_end() -> None:
    workflow = MultiAgentWorkflow(max_iterations=6)
    query = ResearchQuery(query="Research GraphRAG state-of-the-art", max_sources=3)
    state = ResearchState(request=query)

    result = workflow.run(state)

    assert result.final_answer is not None
    assert len(result.sources) > 0
    assert result.analysis_notes is not None
    assert "researcher" in result.route_history
    assert "analyst" in result.route_history
    assert "writer" in result.route_history
    assert result.iteration >= 3


def test_workflow_with_critic() -> None:
    workflow = MultiAgentWorkflow(max_iterations=8, enable_critic=True)
    query = ResearchQuery(query="Compare RAG and fine-tuning for domain adaptation", max_sources=2)
    state = ResearchState(request=query)

    result = workflow.run(state)

    assert result.final_answer is not None
    assert "critic" in result.route_history


def test_workflow_max_iterations_limit() -> None:
    workflow = MultiAgentWorkflow(max_iterations=2)
    query = ResearchQuery(query="Quick test query", max_sources=1)
    state = ResearchState(request=query)

    result = workflow.run(state)
    assert result.iteration <= 3
