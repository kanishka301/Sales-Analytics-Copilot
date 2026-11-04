import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RAW_DIR = ROOT / "data" / "raw"
CALENDAR_CSV = RAW_DIR / "calendar.csv"
PRICES_CSV = RAW_DIR / "sell_prices.csv"
SALES_CSV = RAW_DIR / "sales_train_evaluation.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "m5.duckdb"
DAILY_PARQUET = PROCESSED_DIR / "sales_daily.parquet"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
KNOWLEDGE_DIR = ROOT / "knowledge"

HORIZON = 4
SQL_ROW_LIMIT = 200
SQL_TIMEOUT_S = 20
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_AGENT_STEPS = 5
MAX_TOOL_CHARS = 4000
GEMINI_TIMEOUT_MS = 25000
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "0") == "1"
