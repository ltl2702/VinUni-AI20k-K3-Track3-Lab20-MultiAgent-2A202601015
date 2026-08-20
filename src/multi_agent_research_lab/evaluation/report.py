"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a comprehensive markdown report."""
    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent Systems",
        "",
        "## 1. Quantitative Comparison",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality Score (0-10) | Citation Coverage | Failure Rate | Route History / Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "N/A" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "0%" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        notes = item.notes or "-"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f}s | {cost} | {quality} "
            f"| {citation} | {failure} | `{notes}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Key Insights & Trade-Off Analysis",
            "",
            "- **Chất lượng & Độ chính xác (Quality & Grounding)**:",
            "  - **Multi-agent** đạt điểm chất lượng và độ phủ trích dẫn vượt trội nhờ sự phân tách giữa **Researcher** (thu thập nguồn), **Analyst** (đánh giá phản biện) và **Writer** (tổng hợp có trích dẫn).",
            "  - **Single-agent** trả lời nhanh hơn nhưng không có nguồn tài liệu thực tế dẫn chứng, dễ bị context dilution và hallucination khi xử lý bài toán chuyên sâu.",
            "",
            "- **Độ trễ & Chi phí (Latency & Token Cost)**:",
            "  - **Multi-agent** có độ trễ cao hơn (do phải chạy qua nhiều node tuần tự trong LangGraph) và tiêu tốn nhiều token hơn.",
            "  - **Single-agent** tối ưu cho các tác vụ cần phản hồi tức thì với chi phí tối thiểu.",
            "",
            "- **Khả năng quan sát & Gỡ lỗi (Observability & Debuggability)**:",
            "  - Với **Multi-agent**, mỗi bước thực thi đều được ghi lại trong `route_history` và `trace`, giúp xác định chính xác vị trí phát sinh lỗi.",
            "",
            "## 3. Failure Mode & Mitigation Strategy",
            "",
            "- **Failure Mode:** Vòng lặp vô hạn (Infinite Loop) giữa Supervisor và Worker khi state không được cập nhật đúng hoặc điều kiện dừng bị thiếu.",
            "- **Cách khắc phục (Mitigation):**",
            "  1. Triển khai **Guardrail `max_iterations`** ngắt cưỡng bức khi số vòng lặp vượt ngưỡng cho phép.",
            "  2. Thiết lập cơ chế **Fallback routing** chuyển sang `Writer` hoặc `done` khi gặp lỗi.",
            "  3. Ghi nhận `state.errors` để tracing và cảnh báo kịp thời.",
        ]
    )

    return "\n".join(lines) + "\n"
