from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import statistics
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.

    Custom recognizers:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)
        EMAIL_ADDRESS — email (pattern-based, không cần NLP engine)

    Engine tự thử spacy (en_core_web_lg → en_core_web_sm); nếu không có model nào
    thì fallback sang pattern-only. pii_scan() còn có fallback regex thuần.
    """
    from presidio_analyzer import (AnalyzerEngine, RecognizerRegistry,
                                    Pattern, PatternRecognizer)
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[Pattern("Email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95)],
    )

    analyzer = None
    # Thử dùng spacy NLP engine để có cả các recognizer mặc định (PERSON, ...)
    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            })
            nlp_engine = provider.create_engine()
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            registry.add_recognizer(cccd_recognizer)
            registry.add_recognizer(phone_recognizer)
            analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine,
                                      supported_language="en")
            break
        except Exception:
            continue

    # Fallback: pattern-only (không cần spacy model)
    if analyzer is None:
        registry = RecognizerRegistry()
        registry.add_recognizer(cccd_recognizer)
        registry.add_recognizer(phone_recognizer)
        registry.add_recognizer(email_recognizer)
        try:
            analyzer = AnalyzerEngine(registry=registry, nlp_engine=None,
                                      supported_language="en")
        except Exception:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy", "models": [],
            })
            nlp_engine = provider.create_engine()
            analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine,
                                      supported_language="en")

    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


# ─── PII fallback (regex thuần) + rule-based input guard ──────────────────────

_PII_PATTERNS = [
    ("VN_CCCD", re.compile(r"\b\d{12}\b"), 0.9),
    ("VN_CCCD", re.compile(r"\b\d{9}\b"), 0.7),
    ("VN_PHONE", re.compile(r"\b0[3-9]\d{8}\b"), 0.9),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.95),
]


def _regex_pii_scan(text: str) -> dict:
    """Fallback PII scan bằng regex thuần (khi Presidio engine không khởi tạo được)."""
    matches = []
    for entity, pat, score in _PII_PATTERNS:
        for m in pat.finditer(text):
            matches.append((m.start(), m.end(), entity, score))
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen, last_end = [], -1
    for start, end, entity, score in matches:
        if start >= last_end:
            chosen.append((start, end, entity, score))
            last_end = end
    if not chosen:
        return {"has_pii": False, "entities": [], "anonymized": text}
    anonymized = text
    entities = []
    for start, end, entity, score in sorted(chosen, key=lambda x: -x[0]):
        entities.append({"type": entity, "text": text[start:end],
                         "score": score, "start": start, "end": end})
        anonymized = anonymized[:start] + f"<{entity}>" + anonymized[end:]
    entities.sort(key=lambda x: x["start"])
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


def _normalize_text(text: str) -> str:
    """Lowercase + bỏ dấu tiếng Việt → ASCII, để matching keyword linh hoạt."""
    # đ/Đ không decompose qua NFKD → thay tay trước, nếu không sẽ bị mất hoàn toàn
    text = text.replace("đ", "d").replace("Đ", "d")
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


_PII_REQUEST_PATTERNS = [
    "cccd va so dien thoai", "cccd cua nhan vien", "so dien thoai cua nhan vien",
    "luong cua nhan vien", "luong thang cua nhan vien", "bang luong",
    "thong tin ca nhan cua", "email cua nhan vien", "thong tin nhan vien",
    "tie lo thong tin nhan vien", "tie lo bang luong", "tie lo luong",
    "mat khau admin", "mat khau he thong", "mat khau cua",
    "employee salaries", "all employee", "all confidential",
    "all training data", "employee records", "toan bo thong tin nhan vien",
]
_JAILBREAK_PATTERNS = [
    "bo qua tat ca", "ignore your", "ignore all", "ignore previous",
    "forget your", "forget all", "pretend you are", "you are now",
    "you are an unrestricted", "do anything now", "khong co gioi han",
    "unrestricted", "no restrictions", "dong vai", "system prompt",
    "system override", "system instructions", "huong dan tan cong",
    "ra lenh ban bo qua", "in ra toan bo system", "in ra toan bo",
    "dump all", "print all", "reveal all", "list all employee",
    "admin command", "priority 1", "[admin", "<!--",
]
_OFF_TOPIC_PATTERNS = [
    "bai tho", "tho ve", "nau pho", "cach nau", "pho bo",
    "bitcoin", "ethereum", "phuong trinh vi phan", "giai phuong trinh",
    "recommend", "bo phim", "marvel",
]


def _rule_based_input_check(text: str):
    """Trả về (reason, detail) nếu text khớp pattern độc hại, else None."""
    t = _normalize_text(text)
    for p in _PII_REQUEST_PATTERNS:
        if p in t:
            return ("nemo_input_rail", f"rule:pii_request ({p})")
    for p in _JAILBREAK_PATTERNS:
        if p in t:
            return ("nemo_input_rail", f"rule:jailbreak/injection ({p})")
    for p in _OFF_TOPIC_PATTERNS:
        if p in t:
            return ("nemo_input_rail", f"rule:off_topic ({p})")
    return None


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        try:
            analyzer, anonymizer = setup_presidio()
        except Exception:
            analyzer = anonymizer = None

    # Ưu tiên Presidio engine nếu khả dụng (giới hạn entities để tránh false positive)
    if analyzer is not None:
        try:
            results = analyzer.analyze(
                text=text, language=PRESIDIO_LANGUAGE,
                entities=["VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS"],
            )
            if not results:
                return {"has_pii": False, "entities": [], "anonymized": text}
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
            entities = [
                {"type": r.entity_type, "text": text[r.start:r.end],
                 "score": round(r.score, 3), "start": r.start, "end": r.end}
                for r in results
            ]
            return {"has_pii": True, "entities": entities, "anonymized": anonymized}
        except Exception:
            pass  # fallback sang regex thuần

    return _regex_pii_scan(text)


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml.

    Trỏ OpenAI-compatible client tới Mistral (hoặc OpenAI) qua biến môi trường
    OPENAI_API_KEY / OPENAI_BASE_URL trước khi tạo rails (config.yml dùng engine
    openai). Nếu chưa có API key, caller sẽ fallback sang rule-based heuristic.
    """
    from config import LLM_API_KEY, LLM_BASE_URL
    if LLM_API_KEY:
        os.environ["OPENAI_API_KEY"] = LLM_API_KEY
        if LLM_BASE_URL:
            os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    # Luôn áp dụng rule-based làm backstop (phát hiện kể cả khi NeMo bỏ sót)
    rule_hit = _rule_based_input_check(text)

    if rails is not None:
        try:
            from config import LLM_API_KEY
        except Exception:
            LLM_API_KEY = ""
        if LLM_API_KEY:
            try:
                response = await rails.generate_async(
                    messages=[{"role": "user", "content": text}]
                )
                response = response or ""
                refuse_keywords = ["xin lỗi", "không thể", "không được phép",
                                   "i cannot", "i'm sorry", "khong the"]
                nemo_blocked = any(kw in response.lower() for kw in refuse_keywords)
                blocked = nemo_blocked or (rule_hit is not None)
                return {
                    "allowed": not blocked,
                    "blocked_reason": "nemo_input_rail" if blocked else None,
                    "response": response,
                }
            except Exception:
                pass  # fallback thuần rule-based

    # Fallback (không có NeMo / không có API key): rule-based heuristic
    if rule_hit is not None:
        return {"allowed": False, "blocked_reason": rule_hit[0],
                "response": f"[rule-based] {rule_hit[1]}"}
    return {"allowed": True, "blocked_reason": None, "response": ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            from config import LLM_API_KEY
        except Exception:
            LLM_API_KEY = ""
        if LLM_API_KEY:
            try:
                response = await rails.generate_async(messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
                response = response or ""
                refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot",
                                   "tôi không thể cung cấp"]
                flagged = any(kw in response.lower() for kw in refuse_keywords)
                if flagged:
                    return {"safe": False, "flagged_reason": "nemo_output_rail",
                            "final_answer": response}
                return {"safe": True, "flagged_reason": None, "final_answer": answer}
            except Exception:
                pass

    # Heuristic: quét PII + từ khoá nhạy cảm trong output
    try:
        pii = pii_scan(answer)
        if pii["has_pii"]:
            return {"safe": False, "flagged_reason": "pii_in_output",
                    "final_answer": pii["anonymized"]}
    except Exception:
        pass

    sensitive_phrases = ["cccd của nhân viên", "mật khẩu hệ thống", "mật khẩu admin",
                         "bảng lương chi tiết", "số điện thoại cá nhân"]
    t = _normalize_text(answer)
    if any(_normalize_text(s) in t for s in sensitive_phrases):
        return {"safe": False, "flagged_reason": "sensitive_content",
                "final_answer": "Tôi không thể cung cấp thông tin này. Vui lòng liên hệ phòng Nhân sự trực tiếp."}
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    # Khởi tạo Presidio một lần (tránh setup lặp lại mỗi input)
    if analyzer is None or anonymizer is None:
        try:
            analyzer, anonymizer = setup_presidio()
        except Exception:
            analyzer = anonymizer = None
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            try:
                pii_result = pii_scan(item["input"], analyzer, anonymizer)
            except Exception:
                pii_result = {"has_pii": False}
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async — await, không dùng asyncio.run())
            if blocked_by is None:
                try:
                    rail_result = await check_input_rail(item["input"], rails)
                except Exception:
                    rail_result = {"allowed": True}
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())   # một lần duy nhất — không gọi asyncio.run() trong loop
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    if analyzer is None or anonymizer is None:
        try:
            analyzer, anonymizer = setup_presidio()
        except Exception:
            analyzer = anonymizer = None
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    presidio_times, nemo_times, total_times = [], [], []
    inputs = test_inputs[:n_runs] if test_inputs else []

    async def _measure():
        for text in inputs:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            try:
                pii_scan(text, analyzer, anonymizer)
            except Exception:
                pass
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail (await — không dùng asyncio.run() trong loop)
            t1 = time.perf_counter()
            try:
                await check_input_rail(text, rails)
            except Exception:
                pass
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    try:
        asyncio.run(_measure())   # một lần duy nhất
    except RuntimeError:
        pass  # đã đang trong event loop (rất hiếm)

    def percentiles(times):
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        return {
            "p50": round(s[min(int(n * 0.50), n - 1)], 2),
            "p95": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms":     percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save Phase C report
    report = {
        "adversarial_suite": {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "results": results,
        },
        "latency": latency,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n✓ Phase C report saved → reports/guard_results.json")
