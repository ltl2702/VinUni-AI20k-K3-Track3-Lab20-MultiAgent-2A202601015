"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        try:
            docs = self.search_client.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = docs

            if not docs:
                note = f"No documents found for query: '{state.request.query}'."
                state.research_notes = note
                state.errors.append("Researcher found 0 documents.")
            else:
                lines = [f"### Research Notes for Query: '{state.request.query}'\n"]
                for i, d in enumerate(docs, start=1):
                    url_str = f" ({d.url})" if d.url else ""
                    lines.append(f"[{i}] **{d.title}**{url_str}\n    {d.snippet}")
                state.research_notes = "\n".join(lines)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes or "",
                    metadata={"num_sources": len(docs)},
                )
            )
            state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        except Exception as exc:
            err_msg = f"ResearcherAgent execution error: {exc}"
            state.errors.append(err_msg)
            state.add_trace_event("researcher.error", {"error": str(exc)})

        return state
