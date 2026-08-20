# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

### 1. Case nào NÊN dùng multi-agent? Vì sao?
- **Case nghiên cứu tổng hợp sâu (Deep Research / Complex Synthesis):** Ví dụ như tổng hợp tài liệu học thuật đa nguồn (GraphRAG, khảo sát công nghệ), phân tích báo cáo tài chính doanh nghiệp, hoặc hỗ trợ kỹ thuật khách hàng nhiều cấp độ (Tier-1 Triage -> Specialist -> Resolution).
- **Lý do:**
  - Tách biệt rõ ràng các trách nhiệm (Separation of Concerns): Researcher tập trung truy xuất tài liệu chính xác, Analyst tập trung phản biện và so sánh, Writer tập trung hành văn và trích dẫn chuẩn xác.
  - Ngăn ngừa hiện tượng bão hòa / loãng ngữ cảnh (Context Dilution).
  - Tăng độ phủ trích dẫn (Citation Coverage) và giảm thiểu ảo giác (Hallucination) nhờ cơ chế grounding vào nguồn tài liệu thực tế.
  - Cung cấp khả năng quan sát và gỡ lỗi (Observability) chi tiết tại từng mắt xích thông qua `route_history` và `trace`.

### 2. Case nào KHÔNG NÊN dùng multi-agent? Vì sao?
- **Case tác vụ đơn giản, phản hồi tức thì (Simple Q&A / Low-latency Tasks):** Ví dụ như trả lời câu hỏi trực tiếp (Factual Lookup), dịch thuật đoạn văn ngắn, sinh code snippet đơn giản, hoặc tóm tắt văn bản ngắn.
- **Lý do:**
  - **Độ trễ cao (High Latency):** Multi-agent phải đi qua nhiều lượt gọi LLM tuần tự (Supervisor -> Researcher -> Analyst -> Writer), khiến thời gian phản hồi tăng gấp 3-5 lần.
  - **Chi phí token lớn (High Cost):** Mỗi bước handoff chuyển toàn bộ state/notes qua lại làm tiêu tốn lượng token gấp 2-4 lần so với một prompt đơn lẻ.
  - **Phức tạp hóa không cần thiết (Over-engineering):** Khả năng xảy ra lỗi routing hoặc lỗi đồng bộ state cao hơn trong khi Single-Agent baseline đã đủ giải quyết tốt với chi phí và độ trễ tối ưu.
