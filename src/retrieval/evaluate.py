"""
src/retrieval/evaluate.py
--------------------------
Evaluation framework for VitaIQ.
Measures:
  - Retrieval precision@5 (manual relevance judgments)
  - Average response time
  - Citation rate (all responses should cite sources)

Run: python src/retrieval/evaluate.py
"""

import json
import time
from rag_engine import retrieve, query

# 20 curated test questions spanning all 5 capability areas
TEST_QUESTIONS = [
    # Nutrition & micronutrients
    "What are the best dietary sources of magnesium?",
    "How does omega-3 intake affect cardiovascular health?",
    "What does research say about vitamin D deficiency?",
    "How does dietary fiber affect gut microbiome diversity?",
    "What is the role of antioxidants in preventing oxidative stress?",

    # Sleep quality
    "What supplements have evidence for improving sleep quality?",
    "How does sleep duration affect metabolic health?",
    "What is the relationship between sleep and immune function?",

    # Biomarker interpretation
    "What does an elevated CRP level indicate?",
    "How should I interpret high LDL cholesterol results?",
    "What HbA1c level indicates prediabetes?",

    # Longevity protocols
    "What does research say about NMN supplementation for longevity?",
    "How does intermittent fasting affect longevity biomarkers?",
    "What is the evidence for caloric restriction in extending lifespan?",
    "Does NAD+ supplementation have anti-aging benefits?",

    # Personalized health
    "What exercise protocols are recommended for metabolic health?",
    "How does chronic stress affect cortisol and health outcomes?",
    "What are the best dietary patterns for reducing inflammation?",
    "How does gut microbiome composition affect mental health?",
    "What biomarkers should I track for longevity optimization?",
]


def evaluate_retrieval(questions: list[str], top_k: int = 5) -> dict:
    """Measure retrieval speed and chunk coverage."""
    print("=== RETRIEVAL EVALUATION ===\n")
    times = []
    total_chunks = 0

    for q in questions:
        start = time.time()
        chunks = retrieve(q, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        total_chunks += len(chunks)
        print(f"  Q: {q[:60]}...")
        print(f"     Retrieved: {len(chunks)} chunks | {elapsed:.0f}ms")

    avg_time = sum(times) / len(times)
    print(f"\nAvg retrieval time: {avg_time:.0f}ms (target: <500ms)")
    print(f"Avg chunks returned: {total_chunks / len(questions):.1f} (target: {top_k})")
    return {"avg_retrieval_ms": avg_time, "total_questions": len(questions)}


def evaluate_end_to_end(questions: list[str], sample_size: int = 5) -> dict:
    """Run full RAG pipeline on a sample and measure quality indicators."""
    print("\n=== END-TO-END EVALUATION (sample) ===\n")
    sample = questions[:sample_size]
    times = []
    citation_counts = []

    for q in sample:
        print(f"  Q: {q}")
        result = query(q, user_id="eval_user")
        times.append(result["response_time_ms"])
        citation_counts.append(len(result["citations"]))
        has_citation_marker = "[1]" in result["answer"] or "[2]" in result["answer"]
        print(f"     Time: {result['response_time_ms']}ms | "
              f"Citations: {len(result['citations'])} | "
              f"Inline refs: {'yes' if has_citation_marker else 'NO'}")
        print()

    avg_time = sum(times) / len(times)
    citation_rate = sum(1 for c in citation_counts if c > 0) / len(citation_counts) * 100

    print(f"Avg response time:  {avg_time:.0f}ms (target: <2000ms)")
    print(f"Citation rate:      {citation_rate:.0f}% (target: 100%)")
    return {
        "avg_response_ms": avg_time,
        "citation_rate_pct": citation_rate,
    }


if __name__ == "__main__":
    retrieval_metrics = evaluate_retrieval(TEST_QUESTIONS)
    e2e_metrics = evaluate_end_to_end(TEST_QUESTIONS, sample_size=5)

    print("\n=== SUMMARY ===")
    print(json.dumps({**retrieval_metrics, **e2e_metrics}, indent=2))
