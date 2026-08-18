# 📈 Stock Price Forecasting

Intern ID: CITS8349 
NAME: VAISHNAVI Number of Weeks: 4 
Project Name: Stock -Price Prediction

## Project Scope

### Objective
Build a simple, end-to-end machine learning pipeline that forecasts short-term
stock closing prices using historical price data, as a learning/portfolio
project demonstrating data collection, feature engineering, model training,
evaluation, and visualization.

### In Scope
- Fetching historical daily OHLCV data for a single stock ticker via Yahoo Finance
- Feature engineering using lagged closing prices and rolling statistics
- Training a supervised regression model (Random Forest) to predict next-day close
- Evaluating model performance with MAE and RMSE on a chronological train/test split
- Generating a multi-day (N business days) iterative forecast
- Visualizing historical prices, test predictions, and forecasted prices in a single chart
- Command-line interface for running the pipeline with configurable parameters
  (ticker, history window, forecast horizon, number of lag features)

### Out of Scope
- Real-time or intraday trading signals
- Multi-asset portfolio optimization or backtesting a trading strategy
- Guaranteed predictive accuracy — stock prices are influenced by countless
  unpredictable factors, and short-term movement behaves close to a random walk
- Production deployment (e.g., live API, scheduled retraining, cloud hosting)
- Sentiment analysis, news, or fundamental data integration

### Tech Stack
- **Language:** Python 3.11+ (developed against 3.14)
- **Data source:** `yfinance` (Yahoo Finance

A simple web app that forecasts a stock's next-day closing price using historical
data pulled live from Yahoo Finance (via `yfinance`) and a `RandomForestRegressor`
trained on lag + moving-average features.

Built for internship / portfolio use — clean, small, and easy to explain in an interview.

## How it works
1. You enter a ticker symbol (e.g. `AAPL`, `TCS.NS`, `RELIANCE.NS`).
2. The app downloads ~2 years of daily price history.
3. It engineers features: the last 5 days' closing prices + 5-day and 10-day
   moving averages.
4. It trains a RandomForest model on the first 80% of the data (time-ordered,
   never shuffled) and evaluates it on the most recent 20%.
5. It shows the predicted next-day close, expected change, error metrics
   (RMSE/MAE), and a chart of actual vs predicted prices on the test window.

## Project structure
```
stock-price-forecasting/
├── app.py                # Flask app + model training/inference
├── requirements.txt
├── templates/
│   └── index.html        # UI
└── static/
```

## Run it locally
```bash
git clone https://github.com/<your-username>/stock-price-forecasting.git
cd stock-price-forecasting
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Push it to GitHub
```bash
cd stock-price-forecasting
git init
git add .
git commit -m "Initial commit: stock price forecasting app"
git branch -M main
git remote add origin https://github.com/<your-username>/stock-price-forecasting.git
git push -u origin main
```
Create the empty repo first on github.com (New repository → don't initialize
with a README, since you already have one) and copy its URL for the
`remote add` step above.

## Deploy it live (free) — Render
Render is the easiest free option for a Flask app like this.

1. Push your code to GitHub (steps above).
2. Go to https://render.com and sign in with your GitHub account.
3. Click **New +** → **Web Service** → connect the `stock-price-forecasting` repo.
4. Fill in:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Create Web Service**. Render will build and deploy automatically —
   you'll get a live URL like `https://stock-price-forecasting.onrender.com`.
6. Every future `git push` to `main` auto-redeploys.

(Alternative: Railway, PythonAnywhere, or Streamlit Community Cloud if you
rebuild the UI in Streamlit instead of Flask.)

## Notes for your internship writeup
- **Why RandomForest and not LSTM?** It trains in seconds, needs no GPU, is
  easy to explain (feature importance, no black-box sequence modeling), and
  is a fair baseline before reaching for deep learning. Mentioning this
  tradeoff shows you understand *when* to use simpler models.
- **Time-based train/test split** avoids lookahead bias — a common mistake
  in stock prediction projects where shuffling leaks future prices into training.
- **Possible extensions** to mention if asked: add technical indicators (RSI,
  MACD, Bollinger Bands), try an LSTM/GRU for comparison, add multi-day
  forecasting, or backtest a simple trading strategy against the predictions.

## Disclaimer
This is an educational project. Predictions are not financial advice and
should not be used for real trading decisions.

## Author
Vaishnavi — [GitHub](https://github.com/vshnavi-dot) | [LinkedIn](https://www.linkedin.com/in/vaishnavi-acharya-1955a933b?utm_source=share_via&utm_content=profile&utm_medium=member_android)
