import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

st.set_page_config(page_title="Stonk Forecaster", layout="wide")

MODEL_FACTORIES = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "XGBoost": lambda: XGBRegressor(objective="reg:squarederror", verbosity=0),
    "Neural Network": lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
}
METHODS = ["Recursive", "Date features"]
STRATEGIES = {
    "Follow forecast (long when predicted up)": lambda pr: pr > 0,
    "Contrarian (long when predicted down)": lambda pr: pr < 0,
}
LAGS = 5


@st.cache_data(ttl=600, show_spinner=False)
def get_closes(tickers, start, end):
    data = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True, progress=False, group_by="ticker"
    )
    closes = {}
    for t in tickers:
        try:
            s = data[(t, "Close")]
        except KeyError:
            s = data["Close"]
        s = s.dropna()
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


def forecast_recursive(model_name, close, steps, split):
    df = pd.DataFrame({f"lag_{i}": close.shift(i) for i in range(1, LAGS + 1)})
    df["y"] = close
    df = df.dropna()
    n = int(len(df) * split)
    pipe, ys = train(model_name, df.iloc[:n, :-1], df["y"].iloc[:n].to_numpy())
    test_pred = predict(pipe, ys, df.iloc[n:, :-1])
    err = mape(df["y"].iloc[n:].to_numpy(), test_pred)
    pipe, ys = train(model_name, df.iloc[:, :-1], df["y"].to_numpy())
    cur = close.iloc[-LAGS:].to_numpy()
    preds = np.empty(steps)
    for i in range(steps):
        cur = np.r_[cur[1:], predict(pipe, ys, cur[None])[0]]
        preds[i] = cur[-1]
    return preds, err, df.index[n:], test_pred


def date_features(dates):
    doy = dates.day_of_year.to_numpy() * 2 * np.pi / 366
    return pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "sin_dayofyear": np.sin(doy),
            "cos_dayofyear": np.cos(doy),
            "weekday": dates.day_of_week,
        },
        index=dates,
    )


def forecast_dates(model_name, close, steps, split):
    X = date_features(close.index)
    y = close.to_numpy()
    n = int(len(close) * split)
    pipe, ys = train(model_name, X.iloc[:n], y[:n])
    test_pred = predict(pipe, ys, X.iloc[n:])
    err = mape(y[n:], test_pred)
    future = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=steps)
    return predict(pipe, ys, date_features(future)), err, close.index[n:], test_pred


def strategy_equity(close, test_preds, rule):
    rets = close.pct_change()
    out = {}
    for name, (_, _, tidx, tpred) in test_preds.items():
        pr = pd.Series(tpred, index=tidx) / close.shift(1).reindex(tidx) - 1
        d = pd.concat([rets.reindex(tidx).rename("r"), pr.rename("p")], axis=1).dropna()
        out[name] = (1 + d["r"] * rule(d["p"]).astype(float)).cumprod()
    tidx0 = next(iter(test_preds.values()))[2]
    out["Buy & Hold"] = (1 + rets.reindex(tidx0)).cumprod()
    return pd.DataFrame(out)


def portfolio_equity(equities):
    return pd.concat([e.stack() for e in equities.values()], axis=1).mean(axis=1).unstack()


def render_ticker(t, close, models, methods, steps, split):
    if len(close) < LAGS + 10:
        st.error(f"{t}: not enough data.")
        return
    st.subheader(t)
    future = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=steps)
    results = {}
    for method in methods:
        res = {}
        for name in models:
            if method == "Recursive":
                preds, err, tidx, tpred = forecast_recursive(name, close, steps, split)
            else:
                preds, err, tidx, tpred = forecast_dates(name, close, steps, split)
            res[name] = (preds, err, tidx, tpred)
        results[method] = res
        chart = pd.concat(
            [
                close.rename("Actual"),
                pd.DataFrame({name: v[0] for name, v in res.items()}, index=future),
            ]
        )
        st.write(f"{method} forecast (next {steps} days)")
        st.line_chart(chart)
    return results


st.title("Stock Forecaster")

tickers_input = st.text_input("Stock tickers (comma separated)", "AAPL, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

today = pd.Timestamp.today().date()
c1, c2 = st.columns(2)
start = c1.date_input("Start date", today - pd.Timedelta(days=730))
end = c2.date_input("End date", today)

models = st.multiselect("Models", list(MODEL_FACTORIES), default=list(MODEL_FACTORIES))
methods = st.multiselect("Forecasting methods", METHODS, default=METHODS)
steps = st.slider("Forecast horizon (days)", 1, 90, 10)
split = st.slider("Train split (%)", 50, 95, 80) / 100

if not tickers or not models or not methods:
    st.warning("Enter at least one ticker and pick at least one model and method.")
    st.stop()

closes = get_closes(tuple(tickers), start, end)
if not closes:
    st.error("No data downloaded. Check ticker symbols and date range.")
    st.stop()
missing = [t for t in tickers if t not in closes]
if missing:
    st.warning(f"No data for: {', '.join(missing)}")

all_results = {}
for t, close in closes.items():
    all_results[t] = render_ticker(t, close, models, methods, steps, split)
valid = {t: r for t, r in all_results.items() if r}

if valid:
    with st.expander("Summary", expanded=True):
        method = st.selectbox("Forecasting method", methods)
        rows = []
        for t, close in closes.items():
            last = close.iloc[-1]
            for name, (preds, err, _, _) in valid[t][method].items():
                rows.append(
                    {
                        "Ticker": t,
                        "Model": name,
                        f"Forecast (day {steps})": round(float(preds[-1]), 2),
                        "Change %": round((float(preds[-1]) / last - 1) * 100, 2),
                        "Test MAPE %": round(err * 100, 2),
                    }
                )
        st.dataframe(pd.DataFrame(rows), width="stretch")

    with st.expander("Strategy backtest", expanded=True):
        st.caption(
            "Signals use each model's one-step-ahead out-of-sample test predictions. "
            "Prices are split/dividend adjusted; transaction costs, slippage, and taxes are not modeled."
        )
        level = st.radio("Backtest level", ["Portfolio", "Per stock"], horizontal=True)
        for method in methods:
            st.markdown(f"**{method}**")
            for sname, rule in STRATEGIES.items():
                st.markdown(f"_{sname} — vs Buy & Hold_")
                if level == "Portfolio":
                    eqs = {t: strategy_equity(closes[t], valid[t][method], rule) for t in valid}
                    st.line_chart(portfolio_equity(eqs))
                else:
                    tabs = st.tabs(list(valid))
                    for t, tab in zip(valid, tabs):
                        with tab:
                            st.line_chart(strategy_equity(closes[t], valid[t][method], rule))

    with st.expander("Methodology & references"):
        st.markdown(
            "- Chronological train/test split; all test predictions are one-step-ahead and out-of-sample.\n"
            "- Recursive multi-step forecasts via iterative one-step-ahead prediction "
            "(Bontempi, Ben Taieb & Le Borgne, 2012; Ben Taieb et al., 2012); forecast errors accumulate with horizon.\n"
            "- Calendar-feature models use cyclical encodings but no market information; consistent with weak-form "
            "efficiency, expect weak predictive power (Fama, 1970; Malkiel, 1973).\n"
            "- Inputs and targets are standardized; regularization per Hastie, Tibshirani & Friedman (2009).\n"
            "- MAPE reported per Hyndman & Koehler (2006); asymmetric and ill-defined near zero.\n"
            "- Backtests ignore transaction costs and reflect survivorship bias in yfinance's universe "
            "(Brown, Goetzmann, Ibbotson & Ross, 1992); multiple model-strategy comparisons invite overfitting "
            "caution (Bailey, Borwein, López de Prado & Zhu, 2014).\n"
            "- For an ML-based asset-pricing reference, see Gu, Kelly & Xiu (2020)."
        )
