import os
import sys
import gc
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

# Optimize TensorFlow for low-memory (Render free tier)
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at module level (shared across gunicorn workers)
model_path = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_model.h5')
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path, compile=False)
        model.predict(np.zeros((1, 5)), verbose=0)
        logger.info("Model loaded and warmed up successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
else:
    logger.error("Model file not found at %s", model_path)

tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']
daily_returns_mean = np.array([0.0005, 0.0006, 0.0007, 0.0008, 0.0004])
daily_returns_cov = np.diag([0.0001, 0.00012, 0.00015, 0.0002, 0.0001])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        logger.error("Prediction failed: model not loaded")
        return "Error: Model not loaded", 500

    try:
        user_input = []
        for ticker in tickers:
            val = request.form.get(ticker)
            if val is None or val.strip() == "":
                return f"Error: Missing input for {ticker}", 400
            user_input.append(float(val))

        user_input = np.array(user_input, dtype=np.float32).reshape(1, -1)
        logger.info("Input: %s", user_input.tolist())

        pred_weights = model.predict(user_input, verbose=0, batch_size=1).flatten()
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
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_model.h5')
    if os.path.exists(model_path):
        return "Model exists"
    else:
        return "Model missing"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
