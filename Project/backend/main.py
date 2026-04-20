import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import random
import os
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyswarm import pso
from sklearn.preprocessing import MinMaxScaler

# ===============================
# STABILITY
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

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
# FIXED TOP STOCKS (pre-screened)
# These are well-known high performers
# ===============================
US_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
MY_STOCKS = ["1155.KL", "1023.KL", "5347.KL", "5225.KL", "6033.KL"]

BUDGET = {"US": 2500, "MY": 10000}
CURRENCY = {"US": "USD", "MY": "MYR"}

# ===============================
# STEP 1: FETCH HISTORICAL DATA
# ===============================
def fetch_data(stocks):
    print(f"Fetching data for: {stocks}")
    data = yf.download(stocks, start="2021-01-01", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=stocks[0])
    data = data.ffill().dropna()
    print(f"Data shape: {data.shape}")
    return data

# ===============================
# STEP 2: LSTM PREDICTION
# ===============================
def predict_returns(data, stocks):
    returns = data.pct_change(fill_method=None).dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(returns)

    window = 30  # shorter window = faster
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(len(stocks))
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # Predict next 10 days
    current = scaled[-window:].copy()
    future = []
    for _ in range(10):
        pred = model.predict(current.reshape(1, window, len(stocks)), verbose=0)
        future.append(pred[0])
        current = np.vstack((current[1:], pred))

    expected_returns = np.array(future).mean(axis=0)
    return returns, expected_returns

# ===============================
# STEP 3: PSO WEIGHT OPTIMIZATION
# ===============================
def optimize_weights(returns, stocks):
    def objective(w):
        w = np.array(w)
        w = w / np.sum(w)
        ret = np.sum(returns.mean() * w) * 252
        cov = returns.cov() * 252
        risk = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if risk == 0:
            return 0
        return -(ret / risk)

    w, _ = pso(objective, [0] * len(stocks), [1] * len(stocks),
               swarmsize=20, maxiter=30, debug=False)
    w = np.array(w)
    w = w / np.sum(w)
    return w

# ===============================
# STEP 4: SHARE ALLOCATION
# ===============================
def calculate_allocation(stocks, weights, budget, currency):
    allocation = []
    for i, ticker in enumerate(stocks):
        try:
            price_data = yf.download(ticker, period="2d", auto_adjust=True, progress=False)["Close"]
            price = float(price_data.iloc[-1])
        except Exception:
            price = 0.0

        amount = round(weights[i] * budget, 2)
        shares = int(amount / price) if price > 0 else 0

        allocation.append({
            "ticker": ticker,
            "weight_pct": round(float(weights[i]) * 100, 2),
            "amount": amount,
            "price": round(price, 2),
            "shares": shares
        })

    return allocation

# ===============================
# MAIN PIPELINE
# ===============================
def run_pipeline(market):
    stocks = US_STOCKS if market == "US" else MY_STOCKS
    budget = BUDGET[market]
    currency = CURRENCY[market]

    # 1. Fetch data
    data = fetch_data(stocks)

    # 2. LSTM prediction
    returns, expected_returns = predict_returns(data, stocks)

    # 3. PSO optimization
    weights = optimize_weights(returns, stocks)

    # 4. Allocation
    allocation = calculate_allocation(stocks, weights, budget, currency)

    # Portfolio metrics
    ann_return = float(np.sum(returns.mean() * weights) * 252 * 100)
    cov_matrix = returns.cov() * 252
    ann_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100)
    sharpe = round(ann_return / ann_risk, 4) if ann_risk != 0 else 0

    return {
        "market": market,
        "currency": currency,
        "budget": budget,
        "stocks": stocks,
        "weights": [round(float(w), 4) for w in weights],
        "allocation": allocation,
        "expected_annual_return_pct": round(ann_return, 2),
        "expected_annual_risk_pct": round(ann_risk, 2),
        "sharpe_ratio": sharpe
    }

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
    try:
        return run_pipeline(req.market)
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
