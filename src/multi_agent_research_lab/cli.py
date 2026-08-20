"""Command-line entrypoint for the Multi-Agent Research Lab."""

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    estimate_quality_score,
    run_benchmark,
    run_single_agent,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing()


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""
    _init()
    request = _parse_query(query)
    console.print(f"[bold cyan]Running Single-Agent Baseline for:[/bold cyan] {request.query}\n")

    started = perf_counter()
    state = run_single_agent(request.query)
    latency = perf_counter() - started

    if state.final_answer:
        console.print(Panel(state.final_answer, title="[bold green]Single-Agent Baseline Answer[/bold green]", border_style="green"))

    table = Table(title="Baseline Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Latency", f"{latency:.2f} s")
    table.add_row("Total Input Tokens", str(state.total_input_tokens))
    table.add_row("Total Output Tokens", str(state.total_output_tokens))
    cost_str = f"${state.total_cost_usd:.5f}" if state.total_cost_usd > 0 else "N/A"
    table.add_row("Estimated Cost", cost_str)
    table.add_row("Quality Score", f"{estimate_quality_score(state):.1f}/10")
    table.add_row("Citation Coverage", f"{compute_citation_coverage(state):.0%}")

    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    critic: Annotated[bool, typer.Option("--critic", "-c", help="Enable critic agent")] = False,
) -> None:
    """Run the multi-agent research workflow (Supervisor + Researcher + Analyst + Writer)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow(enable_critic=critic)

    console.print(f"[bold cyan]Launching Multi-Agent Workflow for:[/bold cyan] {request.query}\n")

    started = perf_counter()
    result = workflow.run(state)
    latency = perf_counter() - started

    # Route history
    route_str = " -> ".join(result.route_history) if result.route_history else "None"
    console.print(f"[bold yellow]Route Sequence ({result.iteration} iterations):[/bold yellow] [bold]{route_str}[/bold]\n")

    # Sources
    if result.sources:
        sources_table = Table(title="Retrieved Sources (Researcher)", show_header=True, header_style="bold blue")
        sources_table.add_column("#", width=4)
        sources_table.add_column("Title")
        sources_table.add_column("URL")
        for i, s in enumerate(result.sources, start=1):
            sources_table.add_row(str(i), s.title, s.url or "N/A")
        console.print(sources_table)
        console.print()

    # Final Answer
    if result.final_answer:
        console.print(Panel(result.final_answer, title="[bold green]Final Synthesized Research Report[/bold green]", border_style="green"))
    elif result.errors:
        console.print(Panel(f"Workflow encountered errors: {result.errors}", title="Execution Errors", border_style="red"))

    # Summary table
    table = Table(title="Multi-Agent Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    table.add_row("Latency", f"{latency:.2f} s")
    table.add_row("Total Input Tokens", str(result.total_input_tokens))
    table.add_row("Total Output Tokens", str(result.total_output_tokens))
    cost_str = f"${result.total_cost_usd:.5f}" if result.total_cost_usd > 0 else "N/A"
    table.add_row("Estimated Cost", cost_str)
    table.add_row("Quality Score", f"{estimate_quality_score(result):.1f}/10")
    table.add_row("Citation Coverage", f"{compute_citation_coverage(result):.0%}")
    table.add_row("Errors", str(len(result.errors)))

    console.print(table)


@app.command("benchmark")
def benchmark_cmd(
    output_file: Annotated[str, typer.Option("--output", "-o", help="Output report file")] = "benchmark_report.md",
) -> None:
    """Run benchmark comparing Single-Agent Baseline vs Multi-Agent Workflow."""
    _init()
    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    console.print("[bold green]Starting Benchmark Suite: Single-Agent vs Multi-Agent[/bold green]\n")

    workflow = MultiAgentWorkflow()
    all_metrics = []

    for i, q in enumerate(queries, start=1):
        console.print(f"[bold cyan]Evaluating Query [{i}/{len(queries)}]:[/bold cyan] {q}")

        # Single agent
        _, m_single = run_benchmark(f"Single-Agent (Q{i})", q, run_single_agent)
        all_metrics.append(m_single)

        # Multi agent
        def _run_multi(query_str: str) -> ResearchState:
            req = ResearchQuery(query=query_str)
            st = ResearchState(request=req)
            return workflow.run(st)

        _, m_multi = run_benchmark(f"Multi-Agent (Q{i})", q, _run_multi)
        all_metrics.append(m_multi)

    report_content = render_markdown_report(all_metrics)
    store = LocalArtifactStore()
    saved_path = store.write_text(output_file, report_content)
    console.print(f"\n[bold green]✓ Benchmark complete! Report saved to:[/bold green] {saved_path}")
    console.print(Panel(report_content, title="Benchmark Summary"))


if __name__ == "__main__":
    app()
