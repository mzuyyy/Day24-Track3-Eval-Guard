from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution", "")
        if dist not in groups:  # phòng trường hợp có distribution mới
            groups[dist] = []
        groups[dist].append(item)
    return groups


def _extract_ragas_scores(raw) -> dict:
    """Chuẩn hoá kết quả RAGAS (aggregate dict hoặc per_question list) thành dict 4 metric."""
    defaults = {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0}
    if raw is None:
        return defaults
    # m4_eval có thể trả về {"per_question": [...]}
    if isinstance(raw, dict) and raw.get("per_question"):
        pq = raw["per_question"][0]
        if isinstance(pq, dict):
            return {k: float(pq.get(k, 0.0)) for k in defaults}
        return {k: float(getattr(pq, k, 0.0)) for k in defaults}
    if isinstance(raw, dict):
        return {k: float(raw.get(k, 0.0)) for k in defaults}
    return defaults


def _run_ragas_library(answers: list[dict]) -> list[RagasResult]:
    """Fallback: chạy RAGAS library trực tiếp (cần API key + embeddings model).

    Chỉ dùng khi chưa copy src/m4_eval.py từ Day 18. Best-effort — trả về []
    nếu thiếu dependency hoặc chưa có API key.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  RAGAS library/deps không khả dụng: {e}")
        return []

    if not LLM_API_KEY:
        print("⚠️  Chưa có API key — bỏ qua RAGAS library fallback.")
        return []

    try:
        llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY,
                         base_url=LLM_BASE_URL, temperature=0.0)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        ds = Dataset.from_dict({
            "question": [a["question"] for a in answers],
            "answer": [a["answer"] for a in answers],
            "contexts": [a["contexts"] for a in answers],
            "ground_truth": [a["ground_truth"] for a in answers],
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                                       context_precision, context_recall],
                          llm=llm, embeddings=embeddings)
        df = result.to_pandas()
        results = []
        for i, a in enumerate(answers):
            row = df.iloc[i]
            results.append(RagasResult(
                question_id=a["id"], distribution=a["distribution"],
                question=a["question"], answer=a["answer"],
                contexts=a["contexts"], ground_truth=a["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            ))
        return results
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  RAGAS library fallback lỗi: {e}")
        return []


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Ưu tiên dùng evaluate_ragas() từ Day 18 (src/m4_eval.py). Nếu chưa copy
    Day 18, fallback sang RAGAS library trực tiếp (cần API key + embeddings).
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("⚠️  Không tìm thấy src/m4_eval.py — đã copy từ Day 18 chưa?")
        evaluate_ragas = None

    results: list[RagasResult] = []

    if evaluate_ragas is not None:
        for a in answers:
            try:
                raw = evaluate_ragas(
                    [a["question"]], [a["answer"]],
                    [a["contexts"]], [a["ground_truth"]],
                )
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️  RAGAS lỗi ở câu {a.get('id')}: {e}")
                raw = {}
            scores = _extract_ragas_scores(raw)
            results.append(RagasResult(
                question_id=a["id"], distribution=a["distribution"],
                question=a["question"], answer=a["answer"],
                contexts=a["contexts"], ground_truth=a["ground_truth"],
                faithfulness=scores["faithfulness"],
                answer_relevancy=scores["answer_relevancy"],
                context_precision=scores["context_precision"],
                context_recall=scores["context_recall"],
            ))
        return results

    fallback = _run_ragas_library(answers)
    if fallback:
        return fallback

    print("⚠️  Không thể chạy RAGAS (thiếu Day 18 m4_eval + RAGAS library/API key).")
    return []


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank": i + 1,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "diagnosis": diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        dist = r.distribution
        if dist in matrix.get(r.worst_metric, {}):
            matrix[r.worst_metric][dist] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    if results:
        dominant_dist = max(distributions,
                            key=lambda d: sum(matrix[m][d] for m in matrix))
        dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
        insight = (f"Distribution '{dominant_dist}' có nhiều failure nhất. "
                   f"Metric '{dominant_metric}' là điểm yếu chủ đạo. "
                   f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}")
    else:
        dominant_dist = "n/a"
        dominant_metric = "n/a"
        insight = "Không có dữ liệu để phân tích failure cluster."

    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
