# Design Template: Multi-Agent Research System

## Problem

Hệ thống Research Assistant thông minh tiếp nhận các câu hỏi nghiên cứu chuyên sâu, phức tạp (ví dụ: So sánh công nghệ AI, tổng hợp kỹ thuật GraphRAG, phân tích kiến trúc phân tán), thực hiện tìm kiếm nguồn tài liệu thực tế, phân tích đánh giá chéo và tổng hợp thành báo cáo hoàn chỉnh có cấu trúc và trích dẫn chuẩn xác.

## Why multi-agent?

Khi xử lý bài toán nghiên cứu phức tạp, một Single-Agent (1 prompt duy nhất làm tất cả) thường gặp các vấn đề lớn:
1. **Context Dilution / Saturation**: Một prompt vừa tìm kiếm, vừa lọc nguồn, vừa phân tích phản biện, vừa định dạng văn bản dễ làm mô hình mất tập trung vào các chi tiết quan trọng.
2. **Hallucination & Lack of Grounding**: Không có bước trích xuất và xác thực nguồn độc lập dẫn đến việc LLM tự sinh ra thông tin không có cơ sở hoặc bịa đặt trích dẫn.
3. **Khó Debug & Thiếu Observability**: Khi kết quả cuối cùng bị sai, rất khó xác định lỗi nằm ở khâu tìm kiếm thông tin, khâu suy luận phân tích, hay khâu diễn đạt.

Hệ thống Multi-Agent chia bài toán thành các chuyên viên độc lập: **Supervisor** (điều phối), **Researcher** (thu thập nguồn), **Analyst** (phân tích phản biện), và **Writer** (tổng hợp báo cáo có trích dẫn), phối hợp qua một **Shared State** duy nhất.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Quyết định bước tiếp theo dựa trên state và thực thi guardrails | `ResearchState` | Cập nhật `route_history`, quyết định next node | Lặp vô hạn nếu state không cập nhật |
| **Researcher** | Tìm kiếm tài liệu ngoài qua Search API, thu thập metadata | `request.query`, `max_sources` | `sources`, `research_notes` | Không tìm thấy tài liệu hoặc API lỗi |
| **Analyst** | Phân tích sâu, so sánh các quan điểm, đánh giá độ tin cậy nguồn | `research_notes`, `sources` | `analysis_notes` | Phân tích hời hợt hoặc bỏ sót nguồn |
| **Writer** | Tổng hợp báo cáo chuẩn markdown, gắn trích dẫn `[1]`, `[2]` | `analysis_notes`, `sources`, `audience` | `final_answer` có citations | Trích dẫn sai nguồn hoặc thiếu format |
| **Critic (Optional)** | Phản biện, kiểm tra độ phủ trích dẫn và phát hiện hallucination | `final_answer`, `sources` | Critique feedback, quality score | Đánh giá thiên vị hoặc làm chậm workflow |

## Shared state

Cấu trúc dữ liệu trung tâm `ResearchState`:
- `request: ResearchQuery`: Chứa câu hỏi gốc, số lượng nguồn tối đa và đối tượng độc giả.
- `iteration: int`: Đếm số bước đã chạy, dùng cho guardrail giới hạn vòng lặp.
- `route_history: list[str]`: Ghi nhận lịch sử các agent đã đi qua (`researcher -> analyst -> writer -> done`).
- `sources: list[SourceDocument]`: Danh sách tài liệu thô được tìm kiếm (title, url, snippet, metadata).
- `research_notes: str | None`: Ghi chú tóm tắt từ Researcher.
- `analysis_notes: str | None`: Phân tích so sánh, đánh giá của Analyst.
- `final_answer: str | None`: Báo cáo hoàn chỉnh từ Writer.
- `agent_results: list[AgentResult]`: Kết quả chi tiết và metadata token/cost của từng agent.
- `trace: list[dict]`: Lịch sử sự kiện đo lường thời gian thực thi (observability).
- `errors: list[str]`: Danh sách lỗi phát sinh để xử lý fallback.
- `total_input_tokens`, `total_output_tokens`, `total_cost_usd`: Thống kê tài nguyên tiêu thụ.

## Routing policy

Workflow được xây dựng bằng **LangGraph** theo mô hình Hub-and-Spoke:
```text
               +---------------+
               |     START     |
               +-------+-------+
                       |
                       v
            +----------+----------+
            |      Supervisor     |<-------+
            +----------+----------+        |
                       |                   |
        +--------------+--------------+    |
        |              |              |    |
        v              v              v    |
  +------------+ +------------+ +----------+--+
  | Researcher | |   Analyst  | |   Writer    |
  +-----+------+ +-----+------+ +-----+-------+
        |              |              |
        +--------------+--------------+
                       |
                       v
               (route == "done")
                       v
               +---------------+
               |      END      |
               +---------------+
```

- Nếu `iteration >= max_iterations` -> Chuyển sang `done` (END).
- Nếu `not sources` -> Chuyển sang `researcher`.
- Nếu `sources` và `not analysis_notes` -> Chuyển sang `analyst`.
- Nếu `analysis_notes` và `not final_answer` -> Chuyển sang `writer`.
- Nếu đã có `final_answer` -> Chuyển sang `done` (END).

## Guardrails

- **Max iterations:** Giới hạn tối đa 6-10 vòng lặp, ngăn chặn cháy token khi agent gặp lỗi.
- **Timeout:** 60 giây cho mỗi lượt chạy API / workflow.
- **Retry:** Tự động retry 3 lần với exponential backoff (`tenacity`) khi gặp lỗi mạng/API LLM.
- **Fallback:** Tự động chuyển sang mock response hoặc kết thúc luồng nếu agent thất bại liên tiếp.
- **Validation:** Kiểm tra Pydantic schema cho mọi input/output và xác thực sự hiện diện của citations.

## Benchmark plan

- **Bộ Query đánh giá:**
  1. *"Research GraphRAG state-of-the-art and write a 500-word summary"*
  2. *"Compare single-agent and multi-agent workflows for customer support"*
  3. *"Summarize production guardrails for LLM agents"*
- **Các metric đo lường:**
  - **Latency (s)**: Thời gian hoàn thành từ đầu đến cuối.
  - **Cost (USD)**: Chi phí token theo đơn giá model.
  - **Quality Score (0-10)**: Điểm chất lượng đánh giá độ sâu, cấu trúc và tính đầy đủ.
  - **Citation Coverage (%)**: Tỷ lệ nguồn được trích dẫn chính xác trong câu trả lời.
  - **Failure Rate (%)**: Tỷ lệ câu hỏi không thể hoàn thành hoặc thiếu output.
- **Kết quả kỳ vọng:** Multi-agent đạt điểm chất lượng và độ phủ citation cao hơn 30-50%, đổi lại độ trễ và chi phí cao hơn single-agent.
