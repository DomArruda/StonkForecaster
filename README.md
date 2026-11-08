# Stonk Forecaster

Compare next-day stock-price predictions, then explore portfolio trading based on those predictions.

## Run locally

```bash
python -m uv run streamlit run app.py
```

## Using the app

1. Enter tickers, select models, and click **Run comparison** in the sidebar.
   Expand **Portfolio settings** to choose trading rules, starting investment, and trading costs.
2. **Forecast comparison:** choose a stock. All selected models share one chart with actual prices
   and a yesterday's-close baseline. The accuracy table shows price error and correct up/down predictions.
   Use the dropdown to switch between test-day predictions and future forecasts.
3. **Portfolio strategies:** each model manages the same equal-starting-allocation portfolio using
   each stock's own forecast. Compare returns against buy & hold; choose a trading rule for the graph.
   Rank model/rule combinations by return, risk-adjusted return, or biggest fall from a peak.
4. Expand the full portfolio comparison to see every combination or download a CSV.

Every model uses the five preceding observed closes and the same chronological test dates.
Predictions for a session target its close using earlier observations. Future forecasts are refitted
on all available history, then repeat next-day predictions using their own predicted prices. The portfolio
has equal starting allocations; the modeled trade cost also applies to buy & hold's initial purchase.
“Best” means best on the displayed test period under the selected ranking criterion.
Future dates use weekdays and do not exclude exchange holidays.

## Verify the calculations and UI

```bash
python -m uv run python test_app.py
```

This uses synthetic prices: no market-data connection is needed.

## Dependencies

`pyproject.toml` and `uv.lock` manage the local environment. `requirements.txt` contains the same
locked dependencies for deployment. After changing dependencies, regenerate it with:

```bash
python -m uv export --format requirements.txt --no-hashes --no-emit-project --no-dev --no-annotate --no-header --locked --output-file requirements.txt
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io) → "Create app" → pick repo/branch.
3. Main file path: `app.py`. Select Python 3.14 to match this project's Python requirement.
   Both `uv.lock` and `requirements.txt` are included for dependency installation.
