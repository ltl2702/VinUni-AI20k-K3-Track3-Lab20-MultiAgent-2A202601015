"""Unit tests for agent roles and routing policy."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_supervisor_routing_sequence() -> None:
    supervisor = SupervisorAgent(max_iterations=6)
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG architecture"))

    # Step 1: No sources -> route to researcher
    assert supervisor.decide_route(state) == "researcher"
    state = supervisor.run(state)
    assert state.route_history == ["researcher"]
    assert state.iteration == 1

    # Step 2: Has sources, no analysis -> route to analyst
    state.sources = [
        SourceDocument(title="GraphRAG Overview", snippet="Graph-based retrieval", url="https://example.com")
    ]
    assert supervisor.decide_route(state) == "analyst"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst"]

    # Step 3: Has analysis, no final_answer -> route to writer
    state.analysis_notes = "Detailed comparative analysis."
    assert supervisor.decide_route(state) == "writer"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer"]

    # Step 4: Has final answer -> done
    state.final_answer = "Comprehensive report [1]."
    assert supervisor.decide_route(state) == "done"
    state = supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer", "done"]


def test_supervisor_max_iterations_guardrail() -> None:
    supervisor = SupervisorAgent(max_iterations=3)
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.iteration = 3
    assert supervisor.decide_route(state) == "done"


def test_supervisor_critic_routing() -> None:
    supervisor = SupervisorAgent(max_iterations=6, enable_critic=True)
    state = ResearchState(request=ResearchQuery(query="Test query"))
    state.sources = [SourceDocument(title="Doc 1", snippet="Snippet 1")]
    state.analysis_notes = "Analysis done."
    state.final_answer = "Final answer done."

    # Route should be critic before done
    assert supervisor.decide_route(state) == "critic"


def test_researcher_agent() -> None:
    search_client = SearchClient()
    researcher = ResearcherAgent(search_client=search_client)
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art", max_sources=2))

    updated_state = researcher.run(state)
    assert len(updated_state.sources) > 0
    assert updated_state.research_notes is not None
    assert any(r.agent == AgentName.RESEARCHER for r in updated_state.agent_results)


def test_analyst_agent() -> None:
    llm_client = LLMClient()
    analyst = AnalystAgent(llm_client=llm_client)
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    state.sources = [
        SourceDocument(title="GraphRAG Paper", snippet="Knowledge graphs improve retrieval accuracy.")
    ]
    state.research_notes = "- GraphRAG Paper: Knowledge graphs improve retrieval accuracy."

    updated_state = analyst.run(state)
    assert updated_state.analysis_notes is not None
    assert any(r.agent == AgentName.ANALYST for r in updated_state.agent_results)


def test_writer_agent() -> None:
    llm_client = LLMClient()
    writer = WriterAgent(llm_client=llm_client)
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    state.sources = [
        SourceDocument(title="GraphRAG Paper", snippet="Knowledge graphs improve retrieval.", url="https://arxiv.org/abs/2404.16130")
    ]
    state.analysis_notes = "Analytical synthesis of GraphRAG benefits and trade-offs."

    updated_state = writer.run(state)
    assert updated_state.final_answer is not None
    assert any(r.agent == AgentName.WRITER for r in updated_state.agent_results)


def test_critic_agent() -> None:
    llm_client = LLMClient()
    critic = CriticAgent(llm_client=llm_client)
    state = ResearchState(request=ResearchQuery(query="Research GraphRAG state-of-the-art"))
    state.sources = [SourceDocument(title="GraphRAG Paper", snippet="Knowledge graphs.")]
    state.final_answer = "GraphRAG is a state-of-the-art retrieval method [1]."

    updated_state = critic.run(state)
    assert any(r.agent == AgentName.CRITIC for r in updated_state.agent_results)
