import os
import gc
import logging

os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from flask import Flask, render_template, request
from predict_numpy import predict as model_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Quick warm-up to ensure weights load at startup
model_predict(np.zeros((1, 5), dtype=np.float32))
logger.info("Model loaded and warmed up successfully (NumPy)")

tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']
daily_returns_mean = np.array([0.0005, 0.0006, 0.0007, 0.0008, 0.0004])
daily_returns_cov = np.diag([0.0001, 0.00012, 0.00015, 0.0002, 0.0001])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = []
        for ticker in tickers:
            val = request.form.get(ticker)
            if val is None or val.strip() == "":
                return f"Error: Missing input for {ticker}", 400
            user_input.append(float(val))

        user_input = np.array(user_input, dtype=np.float32).reshape(1, -1)
        logger.info("Input: %s", user_input.tolist())

        pred_weights = model_predict(user_input).flatten()
        logger.info("Raw prediction: %s", pred_weights.tolist())

        s = pred_weights.sum()
        if s <= 0 or np.isnan(s):
            pred_weights = np.full_like(pred_weights, 1.0 / len(tickers))
        else:
            pred_weights = pred_weights / s

        annual_return = float(np.dot(daily_returns_mean, pred_weights) * 252)
        annual_vol = float(np.sqrt(np.dot(pred_weights.T, np.dot(daily_returns_cov * 252, pred_weights))))
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0.0

        portfolio = {tickers[i]: round(float(pred_weights[i]), 4) for i in range(len(tickers))}

        logger.info("Portfolio: %s | Return=%.4f Vol=%.4f Sharpe=%.2f",
                     portfolio, annual_return, annual_vol, sharpe_ratio)

        return render_template('result.html',
                               portfolio=portfolio,
                               annual_return=round(annual_return * 100, 2),
                               annual_vol=round(annual_vol * 100, 2),
                               sharpe_ratio=round(sharpe_ratio, 2))

    except Exception as e:
        logger.error("Prediction error", exc_info=True)
        return f"Prediction error: {str(e)}", 500
    finally:
        gc.collect()

@app.route('/check_model')
def check_model():
    return "Model available (NumPy)" if os.path.exists(
        os.path.join(os.path.dirname(__file__), 'model', 'portfolio_weights.npz')
    ) else "Model weights missing", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
