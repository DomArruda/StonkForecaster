"""Runnable regression check: python -m uv run python test_app.py."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app


class LatestClosePlusOne:
    def predict(self, features):
        return features["lag_1"].to_numpy() + 1


class IdentityScale:
    def inverse_transform(self, values):
        return values


def check():
    # Repeated next-day forecasts must keep the same feature order as training.
    np.testing.assert_allclose(
        app.forecast_next_days(LatestClosePlusOne(), IdentityScale(), pd.Series([20, 30, 40, 50, 60]), 3),
        [61, 62, 63],
    )

    # Irregular trading dates: each forecast earns its target day's return, never tomorrow's.
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"])
    close = pd.Series([100, 110, 99, 108.9, 98.01], index=dates)
    prediction = pd.Series([111, 90, 109, 97], index=dates[1:])
    result = {"Test model": {"test": prediction}}
    rule = app.STRATEGIES["Buy rises / sell falls"]
    equity = app.strategy_equity(close, result, rule)
    assert equity.index.equals(dates)
    np.testing.assert_allclose(equity["Test model"], [1, 1.1, 1.1, 1.21, 1.21])
    np.testing.assert_allclose(equity["Buy & hold"], [1, 1.1, 0.99, 1.089, 0.9801])
    assert app.prediction_stats(close, prediction)["Direction correct (%)"] == 100

    with_costs = app.strategy_equity(close, result, rule, 10)
    np.testing.assert_allclose(with_costs["Test model"].iloc[-1], 1.21 * 0.999 ** 4)
    np.testing.assert_allclose(with_costs["Buy & hold"].iloc[-1], 0.9801 * 0.999)
    assert app.performance_stats(equity["Buy & hold"])["Largest decline (%)"] > 10
    assert app.performance_stats(pd.Series([1, 0.9, 1]))["Largest decline (%)"] > 9.99
    assert np.isnan(app.performance_stats(pd.Series([1, 1, 1]))["Sharpe"])
    neutral = {"Test model": {"test": close.shift(1).loc[dates[1:]].copy()}}
    neutral["Test model"]["test"].iloc[0] = 111
    np.testing.assert_allclose(app.strategy_equity(close, neutral, rule, 10)["Test model"],
                               with_costs["Buy & hold"])
    mismatched = {**result, "Misdated": {"test": prediction.iloc[1:]}}
    try:
        app.strategy_equity(close, mismatched, rule)
        raise AssertionError("Misaligned test dates were accepted")
    except ValueError:
        pass

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=100).delete([8, 35])
    one = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(dates)))), index=dates)
    two = pd.Series(200 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates)))), index=dates)
    models, strategies = list(app.MODEL_FACTORIES), list(app.STRATEGIES)
    analysis = app.analyze({"ONE": one, "TWO": two.iloc[7:]}, models, strategies, 5, 0.8, 10, 10000)
    test_dates = analysis["test_dates"]
    expected_dates = dates[7:][app.LAGS:]
    expected_dates = expected_dates[int(len(expected_dates) * 0.8):]
    assert test_dates.equals(expected_dates)
    comparisons = analysis["comparisons"]
    assert len(comparisons) == len(models) * len(strategies)
    assert "Method" not in comparisons and "Forecast error (%)" not in comparisons
    for criterion, (column, ascending) in app.RANK_BY.items():
        ranked = app.rank_combinations(comparisons, criterion)
        values = ranked[column].dropna()
        assert values.is_monotonic_increasing if ascending else values.is_monotonic_decreasing
    reference_hold = None
    for strategy, curves in analysis["curves"].items():
        assert curves.index[1:].equals(test_dates)
        np.testing.assert_allclose(curves.iloc[0], 1)
        if reference_hold is not None:
            np.testing.assert_allclose(curves["Buy & hold"], reference_hold)
        reference_hold = curves["Buy & hold"]
        stock_curves = {t: app.strategy_equity(analysis["prices"][t], analysis["forecasts"][t],
                                             app.STRATEGIES[strategy], 10) for t in analysis["prices"]}
        np.testing.assert_allclose(curves, (stock_curves["ONE"] + stock_curves["TWO"]) / 2)
    for stock in analysis["prices"]:
        accuracy = analysis["accuracy"][stock]
        assert len(accuracy) == len(models)
        assert accuracy["Forecast error (%)"].is_monotonic_increasing
        frame = app.forecast_lines(analysis, stock, models)
        assert frame.index.equals(test_dates)
        assert set(frame.columns) == {"Actual", "Yesterday's close baseline", *models}
        for model in models:
            np.testing.assert_allclose(frame[model], analysis["forecasts"][stock][model]["test"])
        np.testing.assert_allclose(frame["Yesterday's close baseline"],
                                   analysis["prices"][stock].shift(1).loc[test_dates])
        future = app.forecast_lines(analysis, stock, models, include_future=True)
        assert future.index[-1] > test_dates[-1]
        assert future.loc[future.index > test_dates[-1], "Actual"].isna().all()

    # Changing held-out prices cannot change the first forecast; it should change refitted futures.
    before = app.compute_results(one, ["Linear Regression"], 3, 0.8)
    changed = one.copy()
    heldout = before["Linear Regression"]["test"].index
    changed.loc[heldout] *= 1.1
    after = app.compute_results(changed, ["Linear Regression"], 3, 0.8)
    np.testing.assert_allclose(before["Linear Regression"]["test"].iloc[0],
                               after["Linear Regression"]["test"].iloc[0])
    assert not np.allclose(before["Linear Regression"]["future"], after["Linear Regression"]["future"])

    # Separate views: all models on each chart; browsing must not download or refit anything.
    analysis["missing"] = []
    ui = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30)
    ui.session_state["comparison"] = analysis
    with patch("yfinance.download", side_effect=AssertionError("A graph change downloaded prices")), \
            patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("A graph change retrained a model")):
        ui.run()
        assert not ui.exception and not ui.error
        assert len(ui.metric) == 7 and len(ui.dataframe) == 3
        tabs = ui.get("tab")
        assert [tab.label for tab in tabs] == ["1. Forecast comparison", "2. Portfolio strategies"]
        assert all(sum(e.type.endswith("vega_lite_chart") for e in tab) == 1 for tab in tabs)
        finance_before = ui.dataframe[1].value.copy()
        for stock in analysis["prices"]:
            ui.selectbox(key="forecast_stock").select(stock).run()
            pd.testing.assert_frame_equal(finance_before, ui.dataframe[1].value)
            for view in ui.selectbox(key="forecast_view").options:
                ui.selectbox(key="forecast_view").select(view).run()
                assert not ui.exception and not ui.error
                assert sum(e.type.endswith("vega_lite_chart") for e in ui) == 2
        for criterion in app.RANK_BY:
            forecast_before = ui.dataframe[0].value.copy()
            ui.selectbox(key="portfolio_criterion").select(criterion).run()
            pd.testing.assert_frame_equal(forecast_before, ui.dataframe[0].value)
            winner = app.rank_combinations(comparisons, criterion).iloc[0]
            strategy_key = f"portfolio_strategy_{criterion}"
            assert ui.selectbox(key=strategy_key).value == winner["Trading rule"]
            for strategy in strategies:
                ui.selectbox(key=strategy_key).select(strategy).run()
                assert not ui.exception and not ui.error
        text = " ".join(e.value for kind in ("caption", "markdown", "subheader") for e in ui.get(kind))
        assert "recursive" not in text.lower() and "date features" not in text.lower()
        assert [m.label for m in ui.sidebar.multiselect] == ["Models to compare", "Trading rules"]

    # Submit the actual form, including duplicate/invalid tickers and yfinance's ticker-first columns.
    prices = pd.DataFrame({("ONE", "Close"): one, ("TWO", "Close"): two.iloc[7:],
                           ("BAD", "Close"): np.nan})
    fresh = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
    fresh.sidebar.text_input[0].set_value("ONE, TWO, ONE, BAD")
    fresh.sidebar.multiselect[1].set_value(strategies)
    with patch("yfinance.download", return_value=prices) as download:
        fresh.sidebar.button[0].click().run()
        assert not fresh.exception and not fresh.error
        assert fresh.session_state["comparison"]["missing"] == ["BAD"]
        assert len(fresh.session_state["comparison"]["comparisons"]) == 15
        assert fresh.session_state["comparison"]["cost_bps"] == 10
        assert download.call_args.args[0] == ["ONE", "TWO", "BAD"]
        assert download.call_args.kwargs["end"] == app.date.today() + app.timedelta(days=1)
    print("Passed: date alignment, costs, rankings, all models/rules, and both student-facing views.")


if __name__ == "__main__":
    check()
