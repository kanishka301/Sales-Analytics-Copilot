"""The small set of tools the Gemini agent can use."""
from src.config import ENABLE_WEB_SEARCH


class ToolRegistry:
    def __init__(self, tools):
        self.tools = tools

    def describe(self):
        return "\n".join(
            f"- {name}: {description}" for name, (_, description) in self.tools.items()
        )

    def call(self, name, args):
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}
        if not isinstance(args, dict):
            return {"error": "Tool arguments must be a JSON object."}
        try:
            return self.tools[name][0](**args)
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}


def default_registry():
    from src.tools.analysis_tools import decompose_change, detect_anomalies, price_elasticity
    from src.tools.forecast_tool import backtest_summary, forecast_demand
    from src.tools.sql_tool import run_sql

    tools = {
        "run_sql": (run_sql, "args {query}: run one read-only SELECT on sales_daily."),
        "decompose_change": (decompose_change, "Compare two equal-length periods and explain price, volume and segment contributions."),
        "forecast_demand": (forecast_demand, "Forecast the next 1-4 weeks for a selected state, store, category or department."),
        "backtest_summary": (backtest_summary, "Return saved forecasting accuracy metrics and baseline comparisons."),
        "detect_anomalies": (detect_anomalies, "Find unusual full sales weeks for selected filters and dates."),
        "price_elasticity": (price_elasticity, "Estimate price elasticity as an observational association, not causation."),
    }

    if ENABLE_WEB_SEARCH:
        from src.tools.search_tool import web_search
        tools["web_search"] = (
            web_search,
            "Search external context such as holidays and events; never use for sales figures.",
        )

    return ToolRegistry(tools)
