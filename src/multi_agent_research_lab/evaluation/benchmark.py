"""Benchmark suite for comparing single-agent and multi-agent workflows."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the ratio of retrieved sources referenced in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0

    answer = state.final_answer.lower()
    cited_count = 0

    for i, source in enumerate(state.sources, start=1):
        # Match citation index [1], [2], or title keywords, or url
        idx_pattern = rf"\[{i}\]"
        has_idx = re.search(idx_pattern, state.final_answer) is not None
        has_title = source.title.lower() in answer
        has_url = source.url is not None and source.url.lower() in answer

        if has_idx or has_title or has_url:
            cited_count += 1

    return min(1.0, round(cited_count / len(state.sources), 2))


def estimate_quality_score(state: ResearchState) -> float:
    """Estimate a quality score (0.0 - 10.0) based on structure, depth, citations, and errors."""
    if not state.final_answer:
        return 0.0

    score = 3.0  # Base score for providing an answer

    # Length and depth check
    word_count = len(state.final_answer.split())
    if word_count >= 200:
        score += 2.0
    elif word_count >= 100:
        score += 1.0

    # Formatting structure (headers, bullet points)
    if "#" in state.final_answer and "-" in state.final_answer:
        score += 1.5
    elif "#" in state.final_answer or "-" in state.final_answer:
        score += 1.0

    # Citation coverage reward
    cov = compute_citation_coverage(state)
    score += cov * 2.5

    # Penalize errors
    if not state.errors:
        score += 1.0
    else:
        score = max(0.0, score - min(2.0, len(state.errors) * 0.5))

    return round(min(10.0, score), 1)


def run_single_agent(query: str, llm_client: LLMClient | None = None) -> ResearchState:
    """Single-agent baseline: executes a single LLM prompt without dedicated search/analyst separation."""
    client = llm_client or LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are an AI research assistant. Answer the user query thoroughly in clear markdown "
        "with an overview, main analysis, and conclusion. You do not have access to an external search tool."
    )
    user_prompt = (
        f"Query: {query}\n"
        f"Audience: {request.audience}\n\n"
        "Please provide a comprehensive response."
    )

    response = client.complete(system_prompt, user_prompt)
    state.final_answer = response.content
    state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    state.record_route("single_agent")
    state.add_trace_event("single_agent.done", {"cost_usd": response.cost_usd})

    return state


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute runner, measure latency, token costs, quality, and citation coverage."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cov = compute_citation_coverage(state)
    quality = estimate_quality_score(state)
    cost = state.total_cost_usd if state.total_cost_usd > 0 else None
    failure_rate = 1.0 if (not state.final_answer or len(state.errors) > 0) else 0.0

    notes = ""
    if state.route_history:
        notes = f"Routes: {' -> '.join(state.route_history)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=cov,
        failure_rate=failure_rate,
        notes=notes,
    )
    return state, metrics
