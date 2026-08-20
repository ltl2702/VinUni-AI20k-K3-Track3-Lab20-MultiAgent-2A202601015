"""LangGraph workflow implementation for the multi-agent research system."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and executes the LangGraph multi-agent research graph."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
        max_iterations: int | None = None,
        enable_critic: bool = False,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient()
        settings = get_settings()
        self.max_iterations = max_iterations or settings.max_iterations
        self.enable_critic = enable_critic

        self.supervisor = SupervisorAgent(
            max_iterations=self.max_iterations,
            enable_critic=self.enable_critic,
        )
        self.researcher = ResearcherAgent(search_client=self.search_client)
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)

        self._compiled_graph = None

    def build(self) -> Any:
        """Create and compile the LangGraph state graph."""
        builder = StateGraph(ResearchState)

        # 1. Define nodes
        builder.add_node("supervisor", lambda state: self.supervisor.run(state))
        builder.add_node("researcher", lambda state: self.researcher.run(state))
        builder.add_node("analyst", lambda state: self.analyst.run(state))
        builder.add_node("writer", lambda state: self.writer.run(state))
        if self.enable_critic:
            builder.add_node("critic", lambda state: self.critic.run(state))

        # 2. Entrypoint
        builder.add_edge(START, "supervisor")

        # 3. Conditional routing from supervisor
        def _route_next(state: ResearchState) -> str:
            if not state.route_history:
                return "done"
            last_decision = state.route_history[-1]
            if last_decision in ("researcher", "analyst", "writer", "critic"):
                return last_decision
            return "done"

        route_map: dict[str, str] = {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "done": END,
        }
        if self.enable_critic:
            route_map["critic"] = "critic"

        builder.add_conditional_edges("supervisor", _route_next, route_map)

        # 4. Worker nodes route back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        if self.enable_critic:
            builder.add_edge("critic", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the resulting ResearchState."""
        try:
            if self._compiled_graph is None:
                self._compiled_graph = self.build()

            config = {"recursion_limit": max(25, self.max_iterations * 4)}
            result = self._compiled_graph.invoke(state, config=config)

            if isinstance(result, ResearchState):
                return result
            elif isinstance(result, dict):
                return ResearchState.model_validate(result)
            return state
        except Exception as exc:
            logger.warning("LangGraph invocation encountered error (%s). Using loop execution.", exc)
            return self._run_loop_fallback(state)

    def _run_loop_fallback(self, state: ResearchState) -> ResearchState:
        """Deterministic loop execution fallback matching graph topology."""
        while True:
            state = self.supervisor.run(state)
            next_route = state.route_history[-1] if state.route_history else "done"

            if next_route == "done" or state.iteration >= self.max_iterations:
                break

            if next_route == "researcher":
                state = self.researcher.run(state)
            elif next_route == "analyst":
                state = self.analyst.run(state)
            elif next_route == "writer":
                state = self.writer.run(state)
            elif next_route == "critic":
                state = self.critic.run(state)
            else:
                break

        return state
