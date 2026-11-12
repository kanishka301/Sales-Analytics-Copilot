import json
import re
import time
from dataclasses import dataclass, field

from src.config import MAX_AGENT_STEPS, MAX_TOOL_CHARS

SYSTEM_PROMPT = """You are a sales analytics copilot for the Walmart M5 dataset.
Use only the supplied tool results and schema context for sales figures.

AVAILABLE TOOLS:
{tools}

RELEVANT DATABASE SCHEMA:
{schema}

Return exactly one JSON object:
- To call a tool: {"tool": "tool_name", "args": {}}
- To answer: {"final": "your answer"}

Guidelines:
- For sales totals or factual data, write a read-only SQL query and call run_sql.
- The relevant schema is already supplied above. Do not call a separate schema lookup.
- For changes between periods, use decompose_change.
- For future demand, use forecast_demand.
- For model accuracy, use backtest_summary.
- For unusual weeks, use detect_anomalies.
- For price sensitivity, use price_elasticity.
- Use tool results for every numerical claim. Never invent figures.
- State relevant scope, dates and limitations. Keep answers concise.
"""


@dataclass
class AgentResult:
    ok: bool
    answer: str
    trace: list = field(default_factory=list)
    steps: int = 0


def _parse_action(text):
    """Read one JSON action, allowing a surrounding Markdown code fence."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                break
            except json.JSONDecodeError:
                continue
    if isinstance(value, dict) and ("tool" in value or "final" in value):
        return value
    return None


def _short_result(result):
    text = json.dumps(result, default=str)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + " ...[truncated]"


def run_agent(question, llm=None, registry=None, max_steps=None, history=None):
    """Run a short Gemini -> tool -> Gemini flow and return the answer and trace."""
    if llm is None:
        from src.llm import llm as default_llm
        llm = default_llm
    if registry is None:
        from src.tools.registry import default_registry
        registry = default_registry()

    max_steps = max_steps or MAX_AGENT_STEPS

    # Schema retrieval is local TF-IDF; include it in the first prompt rather than
    # spending an extra Gemini round trip asking the model to look up the schema.
    try:
        from src.tools.schema_rag import lookup_schema
        schema_result = lookup_schema(question, k=4)
        schema_context = "\n\n".join(schema_result.get("context", []))
    except Exception as error:
        schema_context = f"Schema notes unavailable: {type(error).__name__}: {error}"

    system = SYSTEM_PROMPT.replace("{tools}", registry.describe()).replace(
        "{schema}", schema_context or "Use the known sales_daily table and ask for schema only if essential."
    )
    messages = []
    for old_question, old_answer in (history or [])[-3:]:
        messages.extend([
            {"role": "user", "content": old_question},
            {"role": "assistant", "content": json.dumps({"final": old_answer})},
        ])
    messages.append({"role": "user", "content": question})

    trace = []
    seen_calls = set()
    repaired_json = False

    for step in range(1, max_steps + 1):
        started = time.monotonic()
        reply = llm(system, messages) or ""
        llm_seconds = round(time.monotonic() - started, 2)
        messages.append({"role": "assistant", "content": reply})
        action = _parse_action(reply)

        if action is None:
            trace.append({"step": step, "error": "invalid_json", "llm_seconds": llm_seconds,
                          "raw": reply[:250]})
            if repaired_json:
                return AgentResult(False, "Gemini returned an invalid response twice. Please retry.", trace, step)
            repaired_json = True
            messages.append({
                "role": "user",
                "content": 'Your last response was not valid JSON. Return exactly {"tool":"tool_name","args":{}} or {"final":"answer"}.',
            })
            continue

        if "tool" in action:
            name = action.get("tool")
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            call_key = (str(name), json.dumps(args, sort_keys=True, default=str))
            if call_key in seen_calls:
                return AgentResult(
                    False,
                    f"I stopped because the agent repeated the same {name} tool call. Please make the question more specific.",
                    trace,
                    step,
                )
            seen_calls.add(call_key)

            tool_started = time.monotonic()
            result = registry.call(name, args)
            tool_seconds = round(time.monotonic() - tool_started, 2)
            result_text = _short_result(result)
            trace.append({
                "step": step, "tool": name, "args": args, "result": result,
                "is_error": isinstance(result, dict) and "error" in result,
                "llm_seconds": llm_seconds, "tool_seconds": tool_seconds,
            })
            messages.append({
                "role": "user",
                "content": f"TOOL RESULT ({name}):\n{result_text}\n\nUse this result to return the final JSON answer. Call another tool only if essential.",
            })
            continue

        if "final" in action:
            trace.append({"step": step, "llm_seconds": llm_seconds})
            return AgentResult(True, str(action["final"]).strip(), trace, step)

    return AgentResult(
        False,
        f"I could not finish within {max_steps} steps. Try a more specific question.",
        trace,
        max_steps,
    )
