"""Read-only DuckDB access. Every tool uses a cursor so threads (Streamlit) stay safe."""
import duckdb

from src.config import DB_PATH

_con = None


def get_con():
    global _con
    if _con is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(f"{DB_PATH} not found. Run `python -m src.data_prep` first.")
        # read_only: nothing can be modified.
        # enable_external_access=False: SQL cannot read other files on disk (CSV, parquet, ...).
        _con = duckdb.connect(str(DB_PATH), read_only=True, config={"enable_external_access": False})
    return _con


def get_cursor():
    return get_con().cursor()
