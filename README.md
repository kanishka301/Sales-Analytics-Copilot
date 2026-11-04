# Sales Forecasting & Root-Cause Copilot

A Streamlit application for Walmart M5 retail sales analysis. It combines DuckDB SQL analysis, LightGBM demand forecasting, statistical root-cause analysis, schema retrieval, and a Gemini assistant that can call the project's tools.

## Main workflow

1. Prepare the M5 data and DuckDB database.
2. Explore sales patterns with EDA.
3. Train and evaluate forecasting models using rolling-origin backtesting.
4. Ask the Gemini copilot questions about sales, forecasts and drivers.

## Setup (Windows CMD)

```cmd
cd Sales_Copilot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Add your Google AI Studio key to `.env`:

```env
GEMINI_API_KEY=your_key_here
```

Do not share or commit `.env`.

The three M5 source files are included in `data/raw/`. If they are missing, download the M5 Forecasting Accuracy dataset from Kaggle and place `calendar.csv`, `sell_prices.csv`, and `sales_train_evaluation.csv` there.

## Run the project

Run the application from the project root:

```cmd
streamlit run app.py
```

The ZIP already includes the prepared DuckDB database, trained LightGBM model and saved backtest reports. You do not need to rerun data preparation, EDA or model training to use the app.

Only if you intentionally want to rebuild the data and reports, run these separately:

```cmd
python -m src.data_prep
python -m src.eda
python -m src.forecasting.evaluate
```

## How the copilot works

- **Gemini (`src/llm.py`)** sends requests through the Gemini Chat API with a 35-second request timeout and one controlled retry for temporary errors.
- **Agent (`src/agent.py`)** retrieves relevant schema notes locally, asks Gemini which tool is needed, runs it, and asks Gemini for the final explanation. A normal data question therefore needs two Gemini responses rather than a separate schema-lookup round trip.
- **SQL (`src/tools/sql_tool.py`)** runs read-only queries against DuckDB.
- **Schema retrieval (`src/tools/schema_rag.py`)** uses local TF-IDF search to find relevant table definitions and SQL examples. It does not query sales values.
- **Forecasting (`src/tools/forecast_tool.py`)** returns future demand and saved backtest metrics.
- **Root-cause tools (`src/tools/analysis_tools.py` and `stats_tool.py`)** compare periods, detect anomalies and estimate price elasticity.
- **Optional web search** can be enabled with `ENABLE_WEB_SEARCH=1` in `.env` and `pip install ddgs`. External search is for event/holiday context, never sales totals.

Schema retrieval uses TF-IDF, which is sufficient for this small project knowledge base. ChromaDB is not required.

## Forecasting method

The project uses rolling-origin backtesting rather than random train/test splits. It compares LightGBM with simple forecasting baselines using WAPE and skill versus seasonal-naive. Forecasts cover up to four weeks. The saved model and backtest outputs are in `models/` and `reports/`.

## Important limitations

- The M5 sample contains historical sales, calendar and price data, but no inventory, promotion or weather data.
- Forecasts assume recent average prices continue and do not assume future SNAP days or events.
- Price elasticity is an observational association, not proof of causation.
- Anomaly detection can flag normal seasonal or holiday peaks.
- The 80% prediction interval coverage may differ from the nominal 80% target; consult the saved backtest metrics.

## Project structure

```text
app.py                  Streamlit interface
src/agent.py            Gemini tool-calling flow
src/llm.py              Gemini API connection
src/config.py           Paths and settings
src/db.py               Read-only DuckDB connection
src/data_prep.py        Data preparation
src/eda.py              Exploratory data analysis
src/forecasting/        Features, models and backtesting
src/tools/              SQL, schema retrieval, forecasting and statistics
knowledge/              Data dictionary and example SQL
data/raw/               M5 source CSV files
data/processed/         Prepared database and Parquet data
models/                 Trained forecasting model
reports/                Metrics and figures
```
