# Failure Cluster Analysis — Phase A

**Sinh viên:** Chưa cung cấp  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.921 | 1.000 | 0.947 |
| answer_relevancy | 0.705 | 0.640 | 0.667 |
| context_precision | 0.942 | 0.942 | 0.942 |
| context_recall | 0.900 | 0.842 | 0.767 |
| **avg_score** | **0.867** | **0.856** | **0.831** |

Overall weighted avg_score: **0.855** trên 50 câu.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.2965 | context_precision |
| 2 | factual | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.7299 | answer_relevancy |
| 3 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.7333 | context_recall |
| 4 | multi_hop | Nhân viên đi công tác nước ngoài 4 ngày tại thành phố lớn, ở khách sạn 200 USD/đêm (3 đêm). Công ty thanh toán bao nhiêu tiền khách sạn? | 0.7378 | context_recall |
| 5 | multi_hop | Nhân viên tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu: quy trình phê duyệt khác nhau thế nào? | 0.7416 | answer_relevancy |
| 6 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.7425 | answer_relevancy |
| 7 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.7492 | context_precision |
| 8 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | 0.7511 | context_recall |
| 9 | adversarial | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | 0.7843 | context_recall |
| 10 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.7879 | context_recall |

---

## 3. Failure Cluster Matrix

Mỗi ô = số câu có `worst_metric` tương ứng trong distribution đó.

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 1 | 0 | 0 | 1 |
| answer_relevancy | 17 | 15 | 6 | 38 |
| context_precision | 1 | 1 | 1 | 3 |
| context_recall | 1 | 4 | 3 | 8 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

Distribution factual và multi_hop có cùng số lượng câu tuyệt đối trong tập test, nhưng report tự động chọn factual theo tie-break. Điểm yếu chính là `answer_relevancy`: 38/50 câu có metric này là thấp nhất, dù `faithfulness`, `context_precision`, và `context_recall` nhìn chung cao. Điều này cho thấy retrieval thường lấy được tài liệu đúng, nhưng câu trả lời chưa được tổng hợp sát câu hỏi; nhiều answer vẫn giống đoạn context thô hơn là câu trả lời trực tiếp. Với multi-hop, lỗi context_recall tăng lên vì câu hỏi yêu cầu kết hợp nhiều tài liệu hoặc tính toán, trong khi pipeline chỉ lấy top context ngắn.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hoặc fallback answer có thể giữ thông tin ngoài trọng tâm câu hỏi | Siết system prompt: chỉ trả lời từ context, nêu rõ policy version, giảm temperature khi dùng LLM generation |
| context_recall | Thiếu chunk liên quan cho câu hỏi multi-hop/adversarial | Tăng `RERANK_TOP_K`, dùng parent context đầy đủ hơn, bổ sung metadata filter theo version/effective_date |
| context_precision | Có chunk đúng chủ đề nhưng chưa đúng điều kiện/ngưỡng cụ thể | Rerank theo query-aware signals, ưu tiên chunk chứa số tiền/ngày/version khớp câu hỏi |
| answer_relevancy | Answer giống context thô, chưa chuyển thành đáp án ngắn gọn | Bật generation bằng provider thống nhất, prompt yêu cầu trả lời trực tiếp và trích dẫn điều kiện/chính sách liên quan |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial có avg_score **0.831**, thấp hơn factual **0.867** và multi_hop **0.856**. Các câu adversarial trong bottom 10 là Q48, Q41, Q45 và Q46; đây đều là bẫy negation hoặc version conflict. Q41 liên quan nghỉ phép v2023/v2024 và bị context_precision thấp, cho thấy pipeline vẫn retrieve được tài liệu liên quan nhưng còn nhiễu giữa bản cũ và bản hiện hành. Q45, Q46, Q48 chủ yếu yếu ở context_recall hoặc answer_relevancy, nghĩa là cần thêm logic ưu tiên policy hiện hành và prompt trả lời phủ định rõ ràng cho các câu hỏi dạng "có được không".
