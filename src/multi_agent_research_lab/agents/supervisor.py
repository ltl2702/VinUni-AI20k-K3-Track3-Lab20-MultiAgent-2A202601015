"""Supervisor / router agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(
        self,
        max_iterations: int | None = None,
        enable_critic: bool = False,
    ) -> None:
        settings = get_settings()
        self.max_iterations = max_iterations or settings.max_iterations
        self.enable_critic = enable_critic

    def decide_route(self, state: ResearchState) -> str:
        """Determine the next agent route based on current state."""
        # Guardrail: stop if max iterations reached
        if state.iteration >= self.max_iterations:
            return "done"

        # 1. No sources collected yet -> call researcher
        if not state.sources:
            return "researcher"

        # 2. Sources collected, but no analysis notes -> call analyst
        if not state.analysis_notes:
            return "analyst"

        # 3. Analysis done, but no final answer -> call writer
        if not state.final_answer:
            return "writer"

        # 4. Optional critic review
        if self.enable_critic:
            has_critique = any(r.agent == AgentName.CRITIC for r in state.agent_results)
            if not has_critique:
                return "critic"

        # 5. Everything complete -> done
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect state, record routing decision, and return updated state."""
        next_route = self.decide_route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.route",
            {"next": next_route, "iteration": state.iteration},
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routing decision: {next_route}",
                metadata={"next": next_route, "iteration": state.iteration},
            )
        )
        return state
