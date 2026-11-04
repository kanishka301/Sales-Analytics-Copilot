
import json
import time
from pathlib import Path

import pandas as pd

from src.agent import run_agent
from src.db import get_cursor

QUESTIONS = [
    {
        "question": "What is the total number of units sold?",
        "sql": "SELECT SUM(units) FROM sales_daily"
    },
    {
        "question": "What is the total revenue?",
        "sql": "SELECT SUM(revenue) FROM sales_daily"
    },
    {
        "question": "How many distinct stores are present?",
        "sql": "SELECT COUNT(DISTINCT store_id) FROM sales_daily"
    },
    {
        "question": "What is the total revenue in 2015?",
        "sql": "SELECT SUM(revenue) FROM sales_daily WHERE year = 2015"
    },
    {
        "question": "How many units were sold in California?",
        "sql": "SELECT SUM(units) FROM sales_daily WHERE state_id = 'CA'"
    }
]


def evaluate():
    results = []

    for i, q in enumerate(QUESTIONS, 1):
        start = time.time()

        try:
            expected = get_cursor().execute(q["sql"]).fetchone()[0]
            response = run_agent(q["question"])

            answer = response.answer or ""
            numbers = [
                float(x.replace(",", ""))
                for x in __import__("re").findall(
                    r"(?<![\w.])\d[\d,]*(?:\.\d+)?", answer
                )
            ]

            correct = (
                response.ok
                and expected is not None
                and any(
                    abs(n - float(expected)) <= max(0.01, abs(float(expected)) * 0.001)
                    for n in numbers
                )
            )

            elapsed = round(time.time() - start, 2)

            results.append({
                "question": q["question"],
                "expected": expected,
                "answer": answer,
                "correct": correct,
                "time_seconds": elapsed
            })

            print(f"Q{i}: {'PASS' if correct else 'FAIL'}")

        except Exception as e:
            results.append({
                "question": q["question"],
                "expected": None,
                "answer": str(e),
                "correct": False,
                "time_seconds": round(time.time() - start, 2)
            })
            print(f"Q{i}: ERROR - {e}")

    df = pd.DataFrame(results)

    output = Path("reports")
    output.mkdir(exist_ok=True)

    df.to_csv(output / "agent_evaluation.csv", index=False)

    accuracy = df["correct"].mean() * 100
    avg_time = df["time_seconds"].mean()

    summary = {
        "total_questions": len(df),
        "correct_answers": int(df["correct"].sum()),
        "accuracy_percent": round(accuracy, 2),
        "average_response_time_seconds": round(avg_time, 2)
    }

    with open(output / "agent_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nEvaluation Summary")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    evaluate()
