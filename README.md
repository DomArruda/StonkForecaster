# Stonk Forecaster

Streamlit app for stock forecasting: pick tickers, models (Linear/Ridge/Lasso/XGBoost/MLP),
and forecasting methods (recursive or date-features), with strategy backtests vs buy & hold.

## Run locally

```bash
python -m uv run streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io) → "Create app" → pick repo/branch.
3. Main file path: `app.py`. Dependencies are resolved from `pyproject.toml`/`uv.lock` automatically.
