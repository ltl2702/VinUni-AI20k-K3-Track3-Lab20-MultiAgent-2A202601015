"""Critic agent implementation for fact-checking and quality review."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking, citation verification, and hallucination inspection agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append quality critique."""
        if not state.final_answer:
            state.errors.append("CriticAgent: No final_answer available to critique.")
            state.add_trace_event("critic.warning", {"reason": "missing_final_answer"})
            return state

        system_prompt = (
            "You are a rigorous Scientific Peer Reviewer and Fact-Checker. "
            "Evaluate the final answer based on:\n"
            "1. Grounding & Factuality: Are claims supported by the provided research sources?\n"
            "2. Citation Accuracy: Are sources properly cited where needed?\n"
            "3. Completeness & Logical Coherence: Does it directly address the query?\n"
            "Provide a concise evaluation summary, list any potential hallucinations or unsupported claims, "
            "and assign an estimated Quality Score from 0.0 to 10.0."
        )

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{state.research_notes or 'No sources.'}\n\n"
            f"Draft Final Answer:\n{state.final_answer}\n\n"
            "Please provide your critique."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("critic.done", {"cost_usd": response.cost_usd})
        except Exception as exc:
            err_msg = f"CriticAgent execution error: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("critic.error", {"error": str(exc)})

        return state
