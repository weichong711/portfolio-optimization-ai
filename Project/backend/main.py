import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import random

from fastapi import FastAPI
from pydantic import BaseModel
from pyswarm import pso
from sklearn.preprocessing import MinMaxScaler

from supabase import create_client

# ===============================
# STABILITY (VERY IMPORTANT)
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ===============================
# INIT APP
# ===============================
app = FastAPI()

# ===============================
# SUPABASE SETUP
# ===============================
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===============================
# REQUEST MODEL
# ===============================
class RequestModel(BaseModel):
    market: str  # "US" or "MY"

# ===============================
# STOCK LIST
# ===============================
US_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG"]
MY_STOCKS = ["1155.KL", "1023.KL", "5347.KL", "5225.KL", "6033.KL"]

# ===============================
# SAVE RESULT
# ===============================
def save_result(market, result):
    supabase.table("predictions").insert({
        "market": market,
        "stocks": result["stocks"],
        "weights": result["weights"]
    }).execute()

# ===============================
# CORE MODEL
# ===============================
def run_model(stocks):

    # ===== DATA =====
    data = yf.download(stocks, start="2020-01-01", auto_adjust=True)["Close"]
    data = data.ffill().dropna()

    returns = data.pct_change(fill_method=None).dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(returns)

    # ===== SEQUENCE =====
    window = 60
    X, y = [], []

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    # ===== LSTM =====
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(len(stocks))
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    # ===============================
    # STABLE PREDICTION (ENSEMBLE)
    # ===============================
    predictions = []
    last_window = scaled[-60:]

    for _ in range(5):  # ensemble runs
        current = last_window.copy()
        future = []

        for _ in range(30):
            pred = model.predict(current.reshape(1, 60, len(stocks)), verbose=0)
            future.append(pred[0])
            current = np.vstack((current[1:], pred))

        predictions.append(np.array(future))

    predictions = np.mean(predictions, axis=0)
    expected_returns = predictions.mean(axis=0)

    # ===============================
    # STABLE PSO (MULTI RUN)
    # ===============================
    def optimize(selected_returns):

        def objective(w):
            w = np.array(w)
            w = w / np.sum(w)

            ret = np.sum(selected_returns.mean() * w)
            risk = np.sqrt(np.dot(w.T, np.dot(selected_returns.cov(), w)))

            return -(ret / risk)

        results = []

        for _ in range(5):  # multiple PSO runs
            w, _ = pso(
                objective,
                [0]*len(stocks),
                [1]*len(stocks),
                swarmsize=50,
                maxiter=100
            )
            w = w / np.sum(w)
            results.append(w)

        return np.mean(results, axis=0)

    weights = optimize(returns)

    return {
        "stocks": stocks,
        "weights": weights.tolist(),
        "expected_returns": expected_returns.tolist()
    }

# ===============================
# API ROUTE
# ===============================
@app.post("/predict")
def predict(req: RequestModel):

    if req.market == "US":
        result = run_model(US_STOCKS)
    else:
        result = run_model(MY_STOCKS)

    save_result(req.market, result)

    return result
