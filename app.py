"""
Stock Price Forecasting App
----------------------------
A simple Flask web app that:
1. Downloads historical stock price data (via yfinance)
2. Engineers lag + moving-average features
3. Trains a RandomForestRegressor to predict the next day's closing price
4. Shows a chart of actual vs predicted prices + the next-day forecast

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import io
import base64
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed on servers
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from flask import Flask, render_template, request

app = Flask(__name__)

N_LAGS = 5  # how many past days of prices to use as features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features + moving averages from the 'Close' column."""
    data = df.copy()
    data = data[["Close"]].dropna()

    for lag in range(1, N_LAGS + 1):
        data[f"lag_{lag}"] = data["Close"].shift(lag)

    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()

    # target = next day's close
    data["target"] = data["Close"].shift(-1)

    data = data.dropna()
    return data


def train_and_forecast(ticker: str, period: str = "2y"):
    """Download data, train model, evaluate on a held-out tail, and
    forecast the next day's closing price."""
    raw = yf.download(ticker, period=period, progress=False)
    if raw.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # newer yfinance versions can return multi-level columns (ticker, field)
    # even for a single symbol — flatten them if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = build_features(raw)

    feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + ["ma_5", "ma_10"]
    X = data[feature_cols]
    y = data["target"]

    # time-based split: last 20% as test set (never shuffle time series data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    # forecast next day using the most recent available row of features
    last_row = X.iloc[[-1]]
    next_day_pred = float(model.predict(last_row)[0])
    last_close = float(data["Close"].iloc[-1])

    # build chart: actual vs predicted on the test window
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
    ax.plot(y_test.index, preds, label="Predicted", linewidth=2, linestyle="--")
    ax.set_title(f"{ticker.upper()} — Actual vs Predicted Close Price (test set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "ticker": ticker.upper(),
        "last_close": round(last_close, 2),
        "next_day_pred": round(next_day_pred, 2),
        "change": round(next_day_pred - last_close, 2),
        "change_pct": round((next_day_pred - last_close) / last_close * 100, 2),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "plot_b64": plot_b64,
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    ticker = "AAPL"

    if request.method == "POST":
        ticker = request.form.get("ticker", "AAPL").strip()
        try:
            result = train_and_forecast(ticker)
        except Exception as exc:
            error = str(exc)

    return render_template("index.html", result=result, error=error, ticker=ticker)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
