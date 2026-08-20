"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes and sources into structured analytical insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        if not state.sources and not state.research_notes:
            state.errors.append("AnalystAgent: No sources or research notes available to analyze.")
            state.analysis_notes = "No research data available for analysis."
            state.add_trace_event("analyst.warning", {"reason": "missing_sources"})
            return state

        system_prompt = (
            "You are an expert Research Analyst in an advanced multi-agent system. "
            "Your role is to critically analyze the provided research notes and source documents. "
            "1. Extract core claims, mechanisms, and findings.\n"
            "2. Compare perspectives, trade-offs (e.g., latency, cost, complexity, accuracy).\n"
            "3. Assess the credibility and strength of evidence in each source.\n"
            "4. Identify any missing information or critical nuances.\n"
            "Output your analysis in clear, well-structured markdown."
        )

        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            f"Please produce a detailed structured analysis."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst.done",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as exc:
            err_msg = f"AnalystAgent execution error: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("analyst.error", {"error": str(exc)})

        return state
