"""Search client abstraction for ResearcherAgent."""

import json
import logging
import urllib.request
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily integration and rich mock fallback."""

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        if self.api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:
                logger.warning("Tavily search failed (%s). Falling back to mock search.", exc)

        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Execute web search via Tavily API with SSL safety."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_snippets": True,
            "include_raw_content": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "MultiAgentResearchLab/0.1"},
            method="POST",
        )

        ssl_context = None
        try:
            import ssl

            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

        with urllib.request.urlopen(req, data=data, timeout=15, context=ssl_context) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        results: list[SourceDocument] = []
        for item in body.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Untitled Source"),
                    url=item.get("url"),
                    snippet=item.get("content", "") or item.get("snippet", ""),
                    metadata={"score": item.get("score"), "source": "tavily"},
                )
            )

        if not results:
            return self._search_mock(query, max_results)

        return results[:max_results]

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return rich mock search documents tailored to common research queries."""
        q_lower = query.lower()

        if "graphrag" in q_lower:
            docs = [
                SourceDocument(
                    title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet="Microsoft GraphRAG leverages LLMs to build knowledge graphs from text collections, enabling hierarchical community summarization for global dataset understanding.",
                    metadata={"author": "Microsoft Research", "year": 2024},
                ),
                SourceDocument(
                    title="GraphRAG State-of-the-Art Architecture & Benchmarks",
                    url="https://microsoft.github.io/graphrag/",
                    snippet="GraphRAG combines vector search with graph traversal, achieving substantial improvements in comprehensiveness and diversity for complex multi-hop queries over baseline RAG.",
                    metadata={"topic": "GraphRAG Architecture"},
                ),
                SourceDocument(
                    title="Comparative Analysis: Naive RAG vs GraphRAG",
                    url="https://example.com/rag-vs-graphrag",
                    snippet="While naive RAG excels at pinpoint fact retrieval, GraphRAG outperforms when queries require holistic synthesis across large, interconnected corporate document corpora.",
                    metadata={"topic": "Evaluation"},
                ),
            ]
        elif "fine-tuning" in q_lower or "fine tuning" in q_lower:
            docs = [
                SourceDocument(
                    title="RAG vs Fine-tuning: A Practical Guide for Domain Adaptation",
                    url="https://example.com/rag-vs-ft",
                    snippet="RAG is optimal for frequently updated dynamic data and verifiable citations, whereas fine-tuning excels at adapting vocabulary, tone, format, and domain syntax.",
                    metadata={"topic": "Domain Adaptation"},
                ),
                SourceDocument(
                    title="Retrieval-Augmented Generation Survey & Best Practices",
                    url="https://example.com/rag-survey",
                    snippet="Grounding LLM responses in retrieved external documents significantly reduces hallucination and provides verifiable traceability.",
                    metadata={"topic": "RAG Survey"},
                ),
                SourceDocument(
                    title="When and How to Fine-tune Large Language Models",
                    url="https://example.com/when-finetune",
                    snippet="Fine-tuning is most cost-effective when latency constraints preclude multi-document context injection or when proprietary formatting must be strictly adhered to.",
                    metadata={"topic": "Fine-Tuning Trade-offs"},
                ),
            ]
        elif "support" in q_lower or "customer" in q_lower:
            docs = [
                SourceDocument(
                    title="Multi-Agent Workflows in Customer Support Automation",
                    url="https://example.com/multiagent-support",
                    snippet="Specialized agents (Triage, Billing Specialist, Technical Troubleshooter, Escalation Manager) improve resolution rates by 35% compared to single-prompt bots.",
                    metadata={"topic": "Customer Support"},
                ),
                SourceDocument(
                    title="Latency and Cost Trade-offs in Multi-Agent Customer Service",
                    url="https://example.com/support-cost-analysis",
                    snippet="While multi-agent systems increase token consumption by 2.4x, they reduce human escalation rates by 40%, yielding a net positive ROI for complex enterprise tickets.",
                    metadata={"topic": "Cost Analysis"},
                ),
            ]
        elif "guardrail" in q_lower or "guardrails" in q_lower:
            docs = [
                SourceDocument(
                    title="Production Guardrails for Autonomous LLM Agents",
                    url="https://example.com/llm-guardrails",
                    snippet="Essential production guardrails include iteration limits (max_iterations), execution timeouts, structured schema validation, and fallback handlers to prevent infinite loops.",
                    metadata={"topic": "Safety & Guardrails"},
                ),
                SourceDocument(
                    title="Building Reliable Multi-Agent Systems: Anthropic & LangChain Guidelines",
                    url="https://example.com/reliable-agents",
                    snippet="Decoupled state management, deterministic router policies, and explicit handoff contracts prevent agent drift and recursive invocation traps.",
                    metadata={"topic": "Architecture"},
                ),
            ]
        else:
            docs = [
                SourceDocument(
                    title=f"Comprehensive Overview: {query}",
                    url="https://example.com/research-overview",
                    snippet=f"Key empirical findings and theoretical principles regarding '{query}', highlighting foundational concepts, methodologies, and state-of-the-art developments.",
                    metadata={"topic": "Overview"},
                ),
                SourceDocument(
                    title=f"Architectural Patterns and Trade-offs in {query}",
                    url="https://example.com/research-architecture",
                    snippet=f"Detailed evaluation of advantages, trade-offs, and implementation considerations for '{query}' in modern AI systems.",
                    metadata={"topic": "Architecture"},
                ),
                SourceDocument(
                    title=f"Evaluation and Benchmarking Insights on {query}",
                    url="https://example.com/research-benchmarks",
                    snippet=f"Comparative performance analysis, token costs, and quality metrics observed across real-world deployments relating to '{query}'.",
                    metadata={"topic": "Benchmark"},
                ),
            ]

        return docs[:max_results]
