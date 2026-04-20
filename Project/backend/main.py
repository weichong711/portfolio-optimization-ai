import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import random
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyswarm import pso
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client

# ===============================
# STABILITY
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ===============================
# SUPABASE CONFIG
# ===============================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===============================
# FASTAPI + CORS
# ===============================
app = FastAPI(title="Portfolio Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    market: str  # "US" or "MY"

# ===============================
# LARGE STOCK POOLS
# We screen these to find the best 5
# ===============================
US_STOCK_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG",
    "MRK", "ABBV", "LLY", "AVGO", "COST"
]

MY_STOCK_POOL = [
    "1155.KL", "1023.KL", "5347.KL", "5225.KL", "6033.KL",
    "1295.KL", "5183.KL", "6888.KL", "1082.KL", "4197.KL",
    "5285.KL", "6947.KL", "5168.KL", "1066.KL", "2445.KL"
]

# ===============================
# BUDGET CONFIG
# ===============================
BUDGET = {
    "US": 2500,   # USD
    "MY": 10000   # MYR
}

CURRENCY = {
    "US": "USD",
    "MY": "MYR"
}

# ===============================
# STEP 1: SCREEN STOCKS
# Score each stock by Sharpe ratio on historical data
# Pick top 5
# ===============================
def screen_top_stocks(stock_pool, top_n=5):
    print(f"Screening {len(stock_pool)} stocks...")
    try:
        # Download all at once — much faster than one by one
        data = yf.download(stock_pool, start="2022-01-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.ffill().dropna()
        returns = data.pct_change().dropna()

        scores = {}
        for ticker in returns.columns:
            mean_ret = returns[ticker].mean()
            std_ret = returns[ticker].std()
            if std_ret > 0:
                scores[ticker] = mean_ret / std_ret

        top = sorted(scores, key=scores.get, reverse=True)[:top_n]
        print(f"Top {top_n} selected: {top}")
        return top
    except Exception as e:
        print(f"Screening failed: {e}, using defaults")
        return stock_pool[:top_n]

# ===============================
# STEP 2: LSTM PREDICTION
# Predict expected returns for next 30 days
# ===============================
def predict_returns(stocks):
    data = yf.download(stocks, start="2020-01-01", auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.ffill().dropna()
    returns = data.pct_change(fill_method=None).dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(returns)

    window = 60
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(len(stocks))
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # Average over 2 runs for stability
    predictions = []
    last_window = scaled[-60:]

    for _ in range(2):
        current = last_window.copy()
        future = []
        for _ in range(20):
            pred = model.predict(current.reshape(1, 60, len(stocks)), verbose=0)
            future.append(pred[0])
            current = np.vstack((current[1:], pred))
        predictions.append(np.array(future))

    predictions = np.mean(predictions, axis=0)
    expected_returns = predictions.mean(axis=0)

    return returns, expected_returns

# ===============================
# STEP 3: PSO OPTIMIZATION
# Find weights that maximize Sharpe ratio
# ===============================
def optimize_weights(returns, stocks):
    def objective(w):
        w = np.array(w)
        w = w / np.sum(w)
        ret = np.sum(returns.mean() * w) * 252       # annualized
        risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        return -(ret / risk)  # negative because PSO minimizes

    results = []
    for _ in range(2):
        w, _ = pso(objective, [0] * len(stocks), [1] * len(stocks),
                   swarmsize=20, maxiter=30)
        w = w / np.sum(w)
        results.append(w)

    return np.mean(results, axis=0)

# ===============================
# STEP 4: CALCULATE ALLOCATION
# How many shares to buy with the budget
# ===============================
def calculate_allocation(stocks, weights, budget):
    allocation = []
    for i, ticker in enumerate(stocks):
        try:
            price_data = yf.download(ticker, period="1d", auto_adjust=True, progress=False)["Close"]
            price = float(price_data.iloc[-1])
        except Exception:
            price = 0.0

        amount = round(weights[i] * budget, 2)
        shares = int(amount / price) if price > 0 else 0

        allocation.append({
            "ticker": ticker,
            "weight_pct": round(weights[i] * 100, 2),
            "amount": amount,
            "price": round(price, 2),
            "shares": shares
        })

    return allocation

# ===============================
# SAVE TO DATABASE
# ===============================
def save_result(market, result):
    if supabase is None:
        return
    try:
        supabase.table("predictions").insert({
            "market": market,
            "stocks": result["stocks"],
            "weights": result["weights"]
        }).execute()
    except Exception as e:
        print(f"DB save failed: {e}")

# ===============================
# MAIN PIPELINE
# ===============================
def run_pipeline(market):
    pool = US_STOCK_POOL if market == "US" else MY_STOCK_POOL
    budget = BUDGET[market]
    currency = CURRENCY[market]

    # 1. Screen best 5 stocks
    top_stocks = screen_top_stocks(pool, top_n=5)

    # 2. Predict returns with LSTM
    returns, expected_returns = predict_returns(top_stocks)

    # 3. Optimize weights with PSO
    weights = optimize_weights(returns, top_stocks)

    # 4. Calculate share allocation
    allocation = calculate_allocation(top_stocks, weights, budget)

    # Annualized portfolio return & risk
    ann_return = float(np.sum(returns.mean() * weights) * 252 * 100)
    ann_risk = float(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) * 100)
    sharpe = round(ann_return / ann_risk, 4) if ann_risk != 0 else 0

    result = {
        "market": market,
        "currency": currency,
        "budget": budget,
        "stocks": top_stocks,
        "weights": weights.tolist(),
        "allocation": allocation,
        "expected_annual_return_pct": round(ann_return, 2),
        "expected_annual_risk_pct": round(ann_risk, 2),
        "sharpe_ratio": sharpe
    }

    save_result(market, result)
    return result

# ===============================
# API ROUTES
# ===============================
@app.get("/")
def root():
    return {"status": "Portfolio Optimization API is running"}

@app.post("/predict")
def predict(req: RequestModel):
    if req.market not in ["US", "MY"]:
        return {"error": "market must be 'US' or 'MY'"}
    return run_pipeline(req.market)
