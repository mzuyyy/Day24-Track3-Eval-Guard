# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Mai Ngọc Duy - 2A202600736
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~0.73ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~0.78ms P95)
[NeMo Input Rail / rule-based fallback]
    │ block if: off-topic / jailbreak / prompt injection / PII request
    │ action:   return 503 + refusal reason
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Hybrid Search → M3 Rerank → answer generation/fallback
    ▼
[NeMo Output Rail]
    │ flag if: PII in response / sensitive content
    │ action:  redact or replace with safe response
    ▼
User Response
```

---

## Guard Stack Pipeline

| Layer | Tool | Latency P95 | Failure Action |
|---|---|---:|---|
| PII Detection | Presidio + VN regex recognizers | 0.73ms | Reject + log entity type |
| Topic/Jailbreak | NeMo Input Rail + rule-based fallback | 0.78ms | Block + safe refusal |
| RAG Pipeline | Day 18 hybrid RAG | Chưa đo riêng | Fallback to top context / "Không tìm thấy" |
| Output Check | NeMo Output Rail + PII scan fallback | Chưa đo riêng | Redact or block response |
| Total Guard | Presidio + input rail | 1.30ms | Continue only when input is safe |

---

## Latency Budget

Kết quả lấy từ `reports/guard_results.json`. Task 12 hiện đo Presidio và input rail; RAG pipeline và output rail chưa được benchmark riêng trong report này.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| Presidio PII | 0.58 | 0.73 | 0.73 | <10ms |
| NeMo Input Rail | 0.59 | 0.78 | 0.78 | <300ms |
| RAG Pipeline | Chưa đo riêng | Chưa đo riêng | Chưa đo riêng | <2000ms |
| NeMo Output Rail | Chưa đo riêng | Chưa đo riêng | Chưa đo riêng | <300ms |
| **Total Guard** | **1.23** | **1.30** | **1.30** | **<500ms** |

**Budget OK:** Yes  
**Comment:** Guard latency đang rất thấp vì report hiện tại chủ yếu chạy local Presidio/rule-based fallback. Khi bật NeMo LLM rail thực tế qua API, cần đo lại vì latency có thể tăng lên hàng trăm ms và trở thành bottleneck.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65
  pass_if:
    faithfulness: ">= 0.75"
    avg_score: ">= 0.65"

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  pass_if:
    adversarial_suite: ">= 15/20"

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  pass_if:
    total_guard_p95_ms: "< 500"

- name: Judge Reliability Gate
  run: python src/phase_b_judge.py
  pass_if:
    cohen_kappa: ">= 0.60"
```

Kết quả hiện tại:
- RAGAS faithfulness: **0.958** → pass
- RAGAS avg_score: **0.855** → pass
- Adversarial suite pass rate: **20/20** → pass
- Guard P95 latency: **1.30ms** → pass
- Cohen's kappa: **1.000** → pass

---

## Monitoring Dashboard (production)

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS faithfulness (daily sample) | 0.958 | <0.70 | Page on-call, inspect hallucination cases |
| RAGAS answer_relevancy | 0.671 | <0.65 | Review prompt and answer generation |
| Adversarial block rate | 100% | <90% | Review new attack patterns and update rails |
| Guard P95 latency | 1.30ms | >600ms | Profile NeMo/API layer and enable caching |
| PII detected count | N/A | spike >10/hour | Security alert and audit logs |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---:|
| RAGAS avg_score (50q) | 0.855 |
| RAGAS faithfulness | 0.958 |
| RAGAS answer_relevancy | 0.671 |
| RAGAS context_precision | 0.942 |
| RAGAS context_recall | 0.850 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's kappa | 1.000 (almost perfect) |
| Position bias rate | 0.0% |
| Verbosity bias rate | 0.0% |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 1.30ms |

---

## Nhận xét & Cải tiến

Guardrail stack hoạt động tốt trên adversarial suite hiện tại, chặn đủ 20/20 input độc hại và latency local rất thấp. RAGAS cho thấy retrieval khá tốt: `context_precision` 0.942 và `context_recall` 0.850, nhưng `answer_relevancy` chỉ 0.671 nên phần trả lời cần được tổng hợp sát câu hỏi hơn. Nguyên nhân chính là answers hiện tại còn giống top retrieved context, đặc biệt ở các câu hỏi multi-hop và adversarial dạng phủ định/version conflict. Nếu deploy production, cần thống nhất provider LLM cho generation, judge và guardrails; thêm metadata filter cho policy version/effective date; benchmark riêng output rail và RAG pipeline latency trước khi đặt SLO cuối cùng.
