from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

MODEL_FACTORIES = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": lambda: Lasso(alpha=0.01, max_iter=10000),
    "XGBoost": lambda: XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1),
    "Neural Network": lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
}
STRATEGIES = {
    "Buy rises / sell falls": lambda r: np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan)),
    "Buy dips / sell rallies": lambda r: np.where(r < 0, 1.0, np.where(r > 0, 0.0, np.nan)),
    "Buy rises / short falls": lambda r: np.where(r > 0, 1.0, np.where(r < 0, -1.0, np.nan)),
}
STRATEGY_EXPLANATIONS = {
    "Buy rises / sell falls": (
        "**Follow the forecast.** Buy or keep the stock when the model predicts a next-day price above the latest close. "
        "Sell into cash when it predicts a lower price. Cash earns 0% in this comparison."
    ),
    "Buy dips / sell rallies": (
        "**Go against the forecast (contrarian).** Buy when the model predicts a lower next-day price; "
        "sell into cash when it predicts a higher price. This deliberately trades against the model's predicted direction, "
        "rather than buying at an observed future low."
    ),
    "Buy rises / short falls": (
        "**Follow the forecast with shorting.** Buy when the model predicts a higher next-day price and "
        "short when it predicts a lower price. Shorting means betting on a price fall: gains when the stock falls, "
        "losses when it rises. Short-borrow and margin costs are excluded here."
    ),
}
RANK_BY = {
    "Return above buy & hold": ("Vs holding (pp)", False),
    "Risk-adjusted return (Sharpe)": ("Sharpe", False),
    "Smallest decline from peak": ("Largest decline (%)", True),
}
LAGS = 5


@st.cache_data(ttl=600, show_spinner=False)
def get_closes(tickers, start, end):
    data = yf.download(
        list(tickers), start=start, end=end + timedelta(days=1),
        auto_adjust=True, progress=False, group_by="ticker",
    )
    closes = {}
    for t in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            if (t, "Close") not in data.columns:
                continue
            s = data[(t, "Close")]
        elif len(tickers) == 1 and "Close" in data.columns:
            s = data["Close"]
        else:
            continue
        s = pd.to_numeric(s, errors="coerce")
        s = s[np.isfinite(s) & (s > 0)].copy()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if not s.empty:
            closes[t] = s
    return closes


def train(model_name, X, y):
    pipe = make_pipeline(StandardScaler(), MODEL_FACTORIES[model_name]())
    ys = StandardScaler()
    pipe.fit(X, ys.fit_transform(y[:, None]).ravel())
    return pipe, ys


def predict(pipe, ys, X):
    return ys.inverse_transform(pipe.predict(X)[:, None]).ravel()


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def forecast_next_days(pipe, ys, close, steps):
    history = close.iloc[-LAGS:].tolist()
    columns = [f"lag_{i}" for i in range(1, LAGS + 1)]
    forecasts = []
    for _ in range(steps):
        # Training columns are newest to oldest: lag_1, ..., lag_5.
        features = pd.DataFrame([history[-LAGS:][::-1]], columns=columns)
        value = predict(pipe, ys, features)[0]
        forecasts.append(value)
        history.append(value)
    return np.asarray(forecasts)


def compute_results(close, models, steps, split):
    dates = close.index[LAGS:]
    n = int(len(dates) * split)
    if n < 20 or len(dates) - n < 5:
        raise ValueError("Use more history or a smaller training share: need 20 training and 5 test sessions.")
    # ponytail: weekday dates; add exchange calendars for market-specific future holidays.
    future = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=steps)
    X = pd.DataFrame({f"lag_{i}": close.shift(i) for i in range(1, LAGS + 1)}).loc[dates]
    y = close.loc[dates].to_numpy()
    results = {}
    for name in models:
        pipe, ys = train(name, X.iloc[:n], y[:n])
        test = pd.Series(predict(pipe, ys, X.iloc[n:]), index=dates[n:], name=name)
        # Future forecasts use all observed data; test predictions keep the original training fit.
        pipe, ys = train(name, X, y)
        values = forecast_next_days(pipe, ys, close, steps)
        if not np.isfinite(test).all() or not np.isfinite(values).all():
            raise ValueError(f"{name} produced a non-finite forecast. Try a shorter horizon.")
        results[name] = {
            "test": test,
            "future": pd.Series(values, index=future, name=name),
            "model": pipe,
            "y_scaler": ys,
        }
    return results


def prediction_stats(close, prediction):
    actual = close.loc[prediction.index]
    previous = close.shift(1).loc[prediction.index]
    return {
        "Forecast error (%)": mape(actual.to_numpy(), prediction.to_numpy()) * 100,
        "Direction correct (%)": float((np.sign(prediction - previous) ==
                                         np.sign(actual - previous)).mean() * 100),
    }


def strategy_equity(close, test_preds, rule, cost_bps=0.0):
    dates = next(iter(test_preds.values()))["test"].index
    first = close.index.get_loc(dates[0])
    if first == 0:
        raise ValueError("Test forecasts need a preceding observed close.")
    previous = close.shift(1).loc[dates]
    actual_returns = close.loc[dates] / previous - 1
    cost = cost_bps / 10000
    out = {}
    for name, result in test_preds.items():
        prediction = result["test"]
        if not prediction.index.equals(dates):
            raise ValueError("All model predictions must use exactly the same test dates.")
        expected_return = prediction / previous - 1
        position = pd.Series(rule(expected_return), index=dates, dtype=float).ffill().fillna(0)
        turnover = position.diff().abs()
        turnover.iloc[0] = abs(position.iloc[0])
        # ponytail: target-exposure, prior-close fills; use a share/margin ledger for executable trading.
        factors = ((1 - cost * turnover) * (1 + position * actual_returns)).clip(lower=0)
        out[name] = factors.cumprod()
    holding = 1 + actual_returns
    holding.iloc[0] *= 1 - cost  # Same entry cost and mark-to-market ending as active strategies.
    out["Buy & hold"] = holding.cumprod()
    initial = pd.DataFrame(1.0, index=[close.index[first - 1]], columns=out)
    return pd.concat([initial, pd.DataFrame(out)])


def portfolio_equity(equities):
    first = next(iter(equities.values()))
    if any(not e.index.equals(first.index) or not e.columns.equals(first.columns)
           for e in equities.values()):
        raise ValueError("Portfolio comparisons must have the same dates and models for every stock.")
    return sum(equities.values()) / len(equities)


def performance_stats(equity):
    daily = equity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).iloc[1:].fillna(0)
    volatility = daily.std(ddof=1)
    return {
        "Net return (%)": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
        "Largest decline (%)": float((1 - equity / equity.cummax()).max() * 100),
        "Sharpe": float(np.sqrt(252) * daily.mean() / volatility) if volatility > 1e-12 else np.nan,
    }


def analyze(closes, models, strategies, steps, split, cost_bps, capital):
    prices = pd.concat(closes, axis=1, join="inner").dropna().sort_index()
    if prices.empty or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("The selected stocks need overlapping history with positive closing prices.")
    forecasts = {t: compute_results(prices[t], models, steps, split) for t in prices}
    dates = forecasts[prices.columns[0]][models[0]]["test"].index
    accuracy = {
        t: pd.DataFrame([{"Model": model, **prediction_stats(prices[t], forecasts[t][model]["test"])}
                         for model in models]).sort_values(
            ["Forecast error (%)", "Direction correct (%)", "Model"], ascending=[True, False, True],
        ).reset_index(drop=True)
        for t in prices
    }
    curves, stock_curves, rows = {}, {}, []
    for strategy in strategies:
        equities = {t: strategy_equity(prices[t], forecasts[t], STRATEGIES[strategy], cost_bps) for t in prices}
        stock_curves[strategy] = equities
        equity = portfolio_equity(equities)
        curves[strategy] = equity
        holding = performance_stats(equity["Buy & hold"])["Net return (%)"]
        for model in models:
            stats = performance_stats(equity[model])
            rows.append({
                "Combination": f"{model} | {strategy}", "Model": model, "Trading rule": strategy,
                **stats, "Vs holding (pp)": stats["Net return (%)"] - holding,
            })
    naive_errors = {t: prediction_stats(prices[t], prices[t].shift(1).loc[dates])["Forecast error (%)"]
                    for t in prices}
    return {
        "prices": prices, "forecasts": forecasts, "accuracy": accuracy,
        "curves": curves, "stock_curves": stock_curves,
        "comparisons": pd.DataFrame(rows),
        "test_dates": dates, "naive_errors": naive_errors,
        "capital": capital, "cost_bps": cost_bps, "models": list(models),
    }


def rank_combinations(comparisons, criterion):
    column, ascending = RANK_BY[criterion]
    ranked = comparisons.sort_values(
        [column, "Net return (%)", "Combination"],
        ascending=[ascending, False, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def forecast_lines(analysis, stock, models, include_future=False, include_training=False):
    prices = analysis["prices"]
    dates = analysis["test_dates"]
    actual = prices[stock].rename("Actual")
    training_end = prices.index[prices.index.get_loc(dates[0]) - 1]
    anchor = pd.Series(actual.loc[training_end], index=[training_end])
    naive = actual.shift(1).loc[dates]
    lines = [actual]
    for model in models:
        result = analysis["forecasts"][stock][model]
        values = pd.concat([result["test"], result["future"]]) if include_future else result["test"]
        if include_training:
            values = pd.concat([anchor, values])
        lines.append(values.rename(model))
    if include_future:
        future = analysis["forecasts"][stock][models[0]]["future"].index
        naive = pd.concat([naive, pd.Series(actual.iloc[-1], index=future)])
    if include_training:
        naive = pd.concat([anchor, naive])
    lines.append(naive.rename("Yesterday's close baseline"))
    frame = pd.concat(lines, axis=1).sort_index()
    frame.index.name = "Date"
    if include_training:
        return frame
    return frame.loc[prices.index[max(0, len(prices) - 20)]:] if include_future else frame.loc[dates]


def line_figure(frame, y_label):
    data = frame.reset_index()
    figure = px.line(data, x=data.columns[0], y=list(frame.columns), labels={"value": y_label, "variable": ""})
    figure.update_layout(hovermode="x unified", dragmode="zoom", legend_title_text="", height=520)
    figure.update_xaxes(title_text="Date", rangeslider_visible=True)
    return figure


def linear_equation(result):
    pipe, y_scaler = result["model"], result["y_scaler"]
    estimator, x_scaler = pipe[-1], pipe[0]
    coefficients = estimator.coef_ * y_scaler.scale_[0] / x_scaler.scale_
    intercept = (y_scaler.mean_[0] + y_scaler.scale_[0] * float(estimator.intercept_)
                 - float(np.sum(coefficients * x_scaler.mean_)))
    return float(intercept), coefficients


def equation_latex(intercept, coefficients):
    terms = [f"{intercept:.2f}"]
    for i, coefficient in enumerate(coefficients, start=1):
        terms.append(f"{'+' if coefficient >= 0 else '-'} {abs(coefficient):.4f}\\,lag_{i}")
    return r"\text{price}_t = " + " ".join(terms)


def show_equations(analysis, stock):
    with st.expander("Model equations (fitted on all available history)"):
        st.caption("Coefficients are converted back to original price units. The final refit produces the future forecasts; "
                   "test-day accuracy uses the earlier training fit. lag_1 is yesterday's close; lag_5 is five sessions ago.")
        for model in analysis["models"]:
            result = analysis["forecasts"][stock][model]
            estimator = result["model"][-1]
            st.markdown(f"**{model}**")
            if model in ("Linear Regression", "Ridge Regression", "Lasso Regression"):
                st.latex(equation_latex(*linear_equation(result)))
                if model == "Linear Regression":
                    st.caption("Ordinary least squares: minimizes squared error.")
                else:
                    penalty = "L2" if model == "Ridge Regression" else "L1"
                    st.caption(f"{penalty} penalty with alpha = {estimator.alpha:g}, applied to standardized inputs.")
            elif model == "XGBoost":
                trees = estimator.get_booster().trees_to_dataframe()
                gains = trees.groupby("Tree")["Gain"].sum()
                best = int(gains.idxmax())
                st.caption(f"Ensemble of {len(gains)} trees. Predictions add every tree's output plus a base score. "
                           f"Shown: the highest-total-gain tree (tree {best + 1}). "
                           "Features are lag_1 (yesterday's close) through lag_5.")
                tree = trees[trees["Tree"] == best].drop(columns="Tree")
                tree["Feature"] = tree["Feature"].replace({f"f{i}": f"lag_{i + 1}" for i in range(LAGS)})
                st.dataframe(tree, hide_index=True, width="stretch")
            else:
                layers = [estimator.n_features_in_, *estimator.hidden_layer_sizes, 1]
                parameters = sum(w.size for w in estimator.coefs_) + sum(b.size for b in estimator.intercepts_)
                hidden = " → ".join(f"{n} hidden units" for n in layers[1:-1])
                st.markdown(f"Input: {layers[0]} lagged prices → {hidden} → output: 1 price. "
                            f"Activation: {estimator.activation}. {parameters:,} fitted weights and biases.")
                st.caption("A neural network has no compact closed-form equation; the fitted weights define the mapping.")


def show_comparisons(ranked):
    columns = ["Rank", "Model", "Trading rule", "Net return (%)", "Vs holding (pp)",
               "Largest decline (%)", "Sharpe"]
    config = {
        "Net return (%)": st.column_config.NumberColumn("Portfolio return (%)", format="%.2f"),
        "Vs holding (pp)": st.column_config.NumberColumn("Above holding (points)", format="%.2f",
                                                        help="Difference in percentage points, not a relative percentage gain."),
        "Largest decline (%)": st.column_config.NumberColumn("Biggest fall (%)", format="%.2f"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f", help="Return relative to day-to-day risk. Higher is better."),
    }
    st.dataframe(ranked[columns], hide_index=True, width="stretch", column_config=config)


def show_forecasts(analysis):
    st.subheader("How well do the models predict stock prices?")
    c1, c2 = st.columns(2)
    stock = c1.selectbox("Stock to compare", list(analysis["prices"]), key="forecast_stock")
    view = c2.selectbox("Show", ["Predictions vs actual prices", "Future forecasts"], key="forecast_view")
    accuracy = analysis["accuracy"][stock]
    best = accuracy.iloc[0]
    naive_error = analysis["naive_errors"][stock]
    st.markdown(f"**Most accurate model on the test days: {best['Model']}**")
    a, b, c = st.columns(3)
    a.metric("Average price error", f"{best['Forecast error (%)']:.2f}%",
             help="Average absolute percentage error on the test days. Lower is better.")
    b.metric("Up/down predictions correct", f"{best['Direction correct (%)']:.1f}%")
    c.metric("Yesterday's-close forecast error", f"{naive_error:.2f}%")
    if naive_error < accuracy["Forecast error (%)"].min():
        st.info("Predicting the same price as yesterday had a lower price error than every model on these test days.")

    include_training = st.checkbox("Show training history and extend the model lines from it", key="show_training")
    frame = forecast_lines(analysis, stock, analysis["models"], include_future=view == "Future forecasts",
                           include_training=include_training)
    figure = line_figure(frame, "Adjusted closing price")
    for model in analysis["models"]:
        figure.update_traces(line_dash="dash", selector={"name": model})
    prices = analysis["prices"]
    training_end = prices.index[prices.index.get_loc(analysis["test_dates"][0]) - 1]
    figure.add_vline(x=training_end, line_dash="dot", line_color="gray")
    figure.add_annotation(x=training_end, y=1, yref="paper", text="Training ends", showarrow=False, yanchor="bottom")
    if view == "Future forecasts":
        last = prices.index[-1]
        figure.add_vline(x=last, line_dash="dot", line_color="gray")
        figure.add_annotation(x=last, y=1, yref="paper", text="Actual prices end", showarrow=False, yanchor="bottom")
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key="forecast_plot")
    st.caption("Drag to zoom, scroll to zoom further, or use the date slider. Double-click to reset; click legend entries to hide or show models.")
    if include_training:
        st.caption("Dashed model lines join the last training close as a visual starting point. "
                   "Each test-day prediction then uses the preceding observed closes; the join is not an extra scored prediction.")
    if view == "Future forecasts":
        last = analysis["test_dates"][-1]
        st.caption(f"Actual prices end {last:%d %b %Y}. Later values are forecasts made with all available history. "
                   "For each future day, a model uses its own earlier predictions. "
                   "The last 20 observed sessions provide context; future weekday dates may include exchange holidays.")
    else:
        st.caption("All selected models are on the same chart. Each prediction is plotted on the day whose close it predicts, "
                   "using the five preceding observed closes. The yesterday's-close line is a simple comparison forecast.")

    st.markdown("**Accuracy on the test days — lower price error is better**")
    baseline = pd.DataFrame([{"Model": "Yesterday's close (baseline)", "Forecast error (%)": naive_error,
                              "Direction correct (%)": np.nan}])
    table = pd.concat([accuracy, baseline], ignore_index=True).sort_values("Forecast error (%)", kind="stable")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Forecast error (%)": st.column_config.NumberColumn("Average price error (%)", format="%.2f"),
        "Direction correct (%)": st.column_config.NumberColumn("Up/down correct (%)", format="%.1f",
                                                               help="The baseline keeps the price unchanged, so it makes no up/down call."),
    })
    show_equations(analysis, stock)


def show_portfolio(analysis):
    st.subheader("Do the forecasts improve portfolio returns?")
    stocks = ", ".join(analysis["prices"].columns)
    st.caption(f"Portfolio: {stocks}. Starting investment: {analysis['capital']:,.0f}, split equally across the stocks. "
               f"Cost per buy or sell: {analysis['cost_bps'] / 100:.2f}%.")
    criterion = st.selectbox("How should we rank the combinations?", list(RANK_BY),
                             help="Each combination is one model paired with one trading rule. "
                                  "Ranking changes the winner and the table order, not the results.",
                             key="portfolio_criterion")
    ranked = rank_combinations(analysis["comparisons"], criterion)
    best = ranked.iloc[0]
    holding = performance_stats(analysis["curves"][best["Trading rule"]]["Buy & hold"])["Net return (%)"]
    st.markdown(f"**Best combination in this backtest: {best['Model']} · {best['Trading rule']}**")
    a, b, c, d = st.columns(4)
    a.metric("Portfolio return", f"{best['Net return (%)']:+.2f}%", f"{best['Vs holding (pp)']:+.2f} points vs holding")
    b.metric("Buy & hold return", f"{holding:+.2f}%")
    c.metric("Biggest fall from a peak", f"{best['Largest decline (%)']:.2f}%")
    d.metric("Sharpe ratio", "—" if pd.isna(best["Sharpe"]) else f"{best['Sharpe']:.2f}",
             help="Return relative to day-to-day risk. Higher is better.")
    st.caption("These results use the same test-day forecasts shown in Forecast comparison. "
               "The most accurate price model may not produce the best trading return.")
    if ranked["Vs holding (pp)"].max() <= 0:
        st.info("No tested model and trading-rule combination beat simply holding after trading costs.")

    st.markdown("**Portfolio combinations**")
    show_comparisons(ranked)
    st.download_button("Download portfolio comparison CSV", ranked.drop(columns="Combination").to_csv(index=False),
                       file_name="portfolio_comparison.csv", mime="text/csv")

    st.markdown("**Explore a strategy and compare the models**")
    strategies = list(analysis["curves"])
    strategy = st.selectbox("Strategy to explore", strategies, index=strategies.index(best["Trading rule"]),
                            help="Options are the trading rules selected in Portfolio settings.",
                            key=f"portfolio_strategy_{criterion}")
    st.info(STRATEGY_EXPLANATIONS[strategy])
    st.caption("An unchanged forecast leaves the position unchanged. Each model makes its own decisions for each stock.")
    equity = analysis["curves"][strategy]
    figure = line_figure(equity[["Buy & hold", *analysis["models"]]] * analysis["capital"], "Portfolio value")
    st.plotly_chart(figure, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key="portfolio_plot")
    st.caption("Each model trades each stock using that stock's own forecast. All curves start with the same investment; "
               "buy & hold buys at the start and keeps the stocks. This is historical test performance, not projected future profit.")

    with st.expander("Advanced: how each stock contributed"):
        st.caption("Every stock starts with an equal share of the portfolio, and the portfolio return is the average of the "
                   "stocks' returns. A stock's contribution is its own return divided by the number of stocks, so the "
                   "contributions add up to the portfolio return. Contributions are in percentage points of the starting investment.")
        model = st.selectbox("Model to break down", analysis["models"], key=f"contrib_model_{strategy}")
        count = len(analysis["prices"].columns)
        breakdown = pd.DataFrame([
            {
                "Stock": stock,
                "Model return (%)": (equity[model].iloc[-1] - 1) * 100,
                "Portfolio contribution (pp)": (equity[model].iloc[-1] - 1) / count * 100,
                "Buy & hold return (%)": (equity["Buy & hold"].iloc[-1] - 1) * 100,
                "Buy & hold contribution (pp)": (equity["Buy & hold"].iloc[-1] - 1) / count * 100,
            }
            for stock, equity in analysis["stock_curves"][strategy].items()
        ])
        breakdown["Edge vs holding (pp)"] = (breakdown["Portfolio contribution (pp)"]
                                             - breakdown["Buy & hold contribution (pp)"])
        curve = analysis["curves"][strategy]
        total = {
            "Stock": "Total (portfolio)",
            "Model return (%)": (curve[model].iloc[-1] - 1) * 100,
            "Portfolio contribution (pp)": breakdown["Portfolio contribution (pp)"].sum(),
            "Buy & hold return (%)": (curve["Buy & hold"].iloc[-1] - 1) * 100,
            "Buy & hold contribution (pp)": breakdown["Buy & hold contribution (pp)"].sum(),
            "Edge vs holding (pp)": breakdown["Edge vs holding (pp)"].sum(),
        }
        display = pd.concat([breakdown, pd.DataFrame([total])], ignore_index=True)
        st.dataframe(display, hide_index=True, width="stretch",
                     column_config={column: st.column_config.NumberColumn(format="%.2f")
                                    for column in display.columns[1:]})
        bars = breakdown.melt(
            id_vars="Stock",
            value_vars=["Portfolio contribution (pp)", "Buy & hold contribution (pp)"],
            var_name="Measure", value_name="Percentage points",
        )
        figure = px.bar(bars, x="Stock", y="Percentage points", color="Measure", barmode="group",
                        labels={"Measure": ""})
        figure.update_layout(height=360, legend_title_text="")
        st.plotly_chart(figure, width="stretch", config={"displaylogo": False}, key=f"contrib_plot_{strategy}_{model}")
        st.caption("A positive contribution added to the portfolio return; a negative contribution reduced it. "
                   "The edge column shows how much the model's strategy added or lost versus simply holding that stock.")


def main():
    st.set_page_config(page_title="Stock Forecaster", page_icon="capybara.png", layout="wide")
    st.title("Stock Forecaster")
    st.caption("Compare next-day price predictions, then explore what happens when a portfolio trades on those predictions.")

    with st.sidebar.form("settings"):
        st.subheader("Set up your comparison")
        text = st.text_input("Stock tickers, comma separated", "AAPL, MSFT")
        start = st.date_input("History starts", date.today() - timedelta(days=730), max_value=date.today())
        end = st.date_input("History ends", date.today(), max_value=date.today())
        models = st.multiselect("Models to compare", list(MODEL_FACTORIES), default=list(MODEL_FACTORIES))
        steps = st.slider("Days to forecast", 1, 90, 10, help="Weekday dates; exchange holidays are not excluded.")
        split = st.slider("Earlier history used to learn (%)", 50, 95, 80,
                          help="The remaining, later days test predictions against actual prices.") / 100
        with st.expander("Portfolio settings"):
            strategies = st.multiselect("Trading rules", list(STRATEGIES), default=list(STRATEGIES)[:2],
                                       help="Each rule uses each model's own next-day forecast. Selling means cash unless shorting is selected.")
            capital = st.number_input("Starting investment", min_value=100.0, value=10000.0, step=100.0)
            cost_percent = st.number_input("Cost per buy or sell (%)", 0.0, 1.0, 0.10, 0.01,
                                          help="0.10% means a cost of 1 per 1,000 invested. Holding pays the same initial buying cost.")
        submitted = st.form_submit_button("Run comparison", type="primary", width="stretch")

    if submitted:
        tickers = tuple(dict.fromkeys(t.strip().upper() for t in text.split(",") if t.strip()))
        if not tickers or not models or not strategies or start >= end:
            st.error("Choose stocks, models and trading rules, and a start date before the end date.")
            return
        try:
            with st.spinner("Downloading prices, testing forecasts and comparing portfolio results..."):
                closes = get_closes(tickers, start, end)
                if not closes:
                    raise ValueError("No prices downloaded. Check the tickers and date range.")
                analysis = analyze(closes, models, strategies, steps, split, cost_percent * 100, capital)
                analysis["missing"] = [t for t in tickers if t not in closes]
                st.session_state["comparison"] = analysis
        except Exception as exc:
            st.error(f"Could not complete the comparison: {exc}")
            return

    if "comparison" not in st.session_state:
        st.info("Choose your stocks in the sidebar, then click Run comparison.")
        return
    analysis = st.session_state["comparison"]
    if analysis["missing"]:
        st.warning(f"No data for: {', '.join(analysis['missing'])}. Results use the other stocks.")
    dates = analysis["test_dates"]
    st.caption(f"Test days: {dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y} ({len(dates)} trading sessions). "
               "Both views use the same prices, models and next-day test predictions.")
    forecasts_tab, portfolio_tab = st.tabs(["1. Forecast comparison", "2. Portfolio strategies"])
    with forecasts_tab:
        show_forecasts(analysis)
    with portfolio_tab:
        show_portfolio(analysis)

    with st.expander("How to read these results · methodology & references"):
        st.markdown(
            "- **Net return** is the test-period gain after modeled trading costs. **Vs holding** is the difference in percentage points.\n"
            "- **Largest decline** is the maximum drawdown, including the initial investment. Lower is better.\n"
            "- **Sharpe** uses daily net returns, a 0% risk-free rate and 252 sessions per year; zero volatility displays as unavailable.\n"
            "- **Forecast error** is MAPE; **direction correct** compares the predicted and actual move from the preceding close. "
            "Accuracy is evaluated separately for each stock.\n"
            "- All models use the five preceding observed closes and the same chronological training/test targets and common stock dates. "
            "Test predictions target the next observed close. Future forecasts repeat these predictions with no fresh prices. "
            "Trading rules do not change forecast accuracy.\n"
            "- Portfolio sleeves receive equal starting capital and trade independently; weights can drift. Cash earns 0%. "
            "Closing-price fills are hypothetical. Costs apply on position changes; open positions are valued, not liquidated, at the end. "
            "Taxes, intraday slippage and short-borrow/margin costs are excluded.\n"
            "- Each stock's contribution is its own return divided by the number of stocks, so contributions add up to the portfolio return. "
            "The edge column compares each stock's contribution with simply holding that stock.\n"
            "- These rankings select a winner on the displayed test data; independent validation is needed to assess the selected winner.\n\n"
            "References: Hyndman & Athanasopoulos, [forecast benchmarks and accuracy](https://otexts.com/fpp3/toolbox.html#sec-accuracy); "
            "Sharpe (1994), [The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)."
        )


if __name__ == "__main__":
    main()
