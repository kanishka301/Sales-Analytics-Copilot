import pandas as pd
import streamlit as st

from src.agent import run_agent

st.set_page_config(page_title="Sales Analytics Copilot", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting & Root-Cause Copilot")

EXAMPLES = [
    "What were total units and revenue in 2015?",
    "Why did revenue change between January 2015 and January 2016 for California?",
    "Forecast the next 4 weeks of FOODS demand in Texas.",
    "How accurate is the forecasting model?",
    "Which department is most sensitive to price?",
    "Were there unusual sales weeks for store WI_1 in 2014?",
]

with st.sidebar:
    st.subheader("Settings")
    st.subheader("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch"):
            st.session_state["pending"] = ex
    if st.button("Clear chat"):
        st.session_state["history"] = []

st.session_state.setdefault("history", [])


def render_trace(trace):
    for t in trace:
        if "error" in t and "tool" not in t:
            st.warning(f"Step {t['step']}: {t['error']}")
            continue
        icon = "⚠️" if t.get("is_error") else "🔧"
        with st.expander(f"{icon} Step {t['step']}: {t['tool']}"):
            timing = []
            if "llm_seconds" in t:
                timing.append(f"Gemini: {t['llm_seconds']}s")
            if "tool_seconds" in t:
                timing.append(f"Tool: {t['tool_seconds']}s")
            if timing:
                st.caption(" · ".join(timing))
            if t.get("thought"):
                st.caption(t["thought"])
            st.json(t.get("args", {}))
            res = t["result"]
            if isinstance(res, dict) and res.get("rows") is not None and res.get("columns"):
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), width="stretch")
            else:
                st.json(res)


def forecast_chart(trace):
    for t in trace:
        res = t.get("result") or {}
        if t.get("tool") == "forecast_demand" and "forecast" in res:
            hist = pd.Series(res["recent_actual_weekly_units"], name="actual")
            fc = pd.DataFrame(res["forecast"]).set_index("week_start")
            df = pd.concat([hist, fc["forecast_units"].rename("forecast"),
                            fc["lower_80"].rename("lower_80"), fc["upper_80"].rename("upper_80")], axis=1)
            st.line_chart(df)
            return


for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            forecast_chart(msg["trace"])
            render_trace(msg["trace"])

def previous_turns():
    """(question, answer) pairs of earlier turns that finished properly, so follow-ups work."""
    msgs, pairs = st.session_state["history"], []
    for q, a in zip(msgs[0::2], msgs[1::2]):
        if a.get("ok"):
            pairs.append((q["content"], a["content"]))
    return pairs


question = st.chat_input("Ask about sales, drivers, or forecasts...") or st.session_state.pop("pending", None)
if question:
    turns = previous_turns()          # computed before the new question is added
    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                r = run_agent(question, history=turns)
                answer, trace, ok = r.answer, r.trace, r.ok
            except Exception as e:  # missing data, missing API key, etc.
                answer, trace, ok = f"Something went wrong: `{type(e).__name__}: {e}`", [], False
        st.markdown(answer)
        forecast_chart(trace)
        render_trace(trace)
    st.session_state["history"].append(
        {"role": "assistant", "content": answer, "trace": trace, "ok": ok})
