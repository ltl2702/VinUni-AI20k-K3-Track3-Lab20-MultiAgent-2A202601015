"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        sources_list = []
        for i, s in enumerate(state.sources, start=1):
            url_part = f" ({s.url})" if s.url else ""
            sources_list.append(f"[{i}] {s.title}{url_part}: {s.snippet}")
        sources_text = "\n".join(sources_list) if sources_list else "No external sources."

        system_prompt = (
            "You are a Senior Technical Writer in a research laboratory. "
            "Your task is to write a comprehensive, highly accurate, and engaging research synthesis "
            f"tailored for '{state.request.audience}'.\n\n"
            "Requirements:\n"
            "1. Synthesize insights from the provided Analysis Notes and Research Notes.\n"
            "2. Structure with clear Markdown headers, bullet points, and comparative explanations.\n"
            "3. Ground all factual assertions and include citations in the text formatted as [1], [2], etc.\n"
            "4. Include a dedicated '### Nguồn tham khảo (Citations)' section at the end matching the source numbers.\n"
            "5. Maintain a professional, objective, and authoritative tone."
        )

        user_prompt = (
            f"Topic / Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"=== Available Sources ===\n{sources_text}\n\n"
            f"=== Analysis Notes ===\n{state.analysis_notes or state.research_notes or 'No analysis notes.'}\n\n"
            "Please write the complete final research report."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            content = response.content

            # Ensure citations section is present if sources exist and model didn't add it
            if state.sources and "### Nguồn tham khảo" not in content and "### References" not in content:
                citation_lines = ["\n\n### Nguồn tham khảo (Citations)"]
                for i, s in enumerate(state.sources, start=1):
                    url_str = f" ({s.url})" if s.url else ""
                    citation_lines.append(f"[{i}] {s.title}{url_str}")
                content += "\n".join(citation_lines)

            state.final_answer = content
            state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "writer.done",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        except Exception as exc:
            err_msg = f"WriterAgent execution error: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("writer.error", {"error": str(exc)})

        return state
