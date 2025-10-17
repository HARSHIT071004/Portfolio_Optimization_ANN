from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

# Optimize TensorFlow for low-memory (Render free tier)
tf.config.set_visible_devices([], 'GPU')  # disable GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Enable logging to trace requests
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load trained ANN model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_model.h5')
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    logging.info(f"Model loaded successfully from {model_path}")
else:
    model = None
    logging.error("Model file not found!")

tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not found"
    try:
        # Capture input safely
        user_input = []
        for ticker in tickers:
            val = request.form.get(ticker)
            if val is None or val.strip() == "":
                return f"Error: Missing input for {ticker}"
            user_input.append(float(val))
        user_input = np.array(user_input).reshape(1, -1)

        logging.info("Form data received: %s", request.form)
        logging.info("User input array: %s", user_input)

        # Predict (lightweight)
        pred_weights = model.predict(user_input, verbose=0).flatten()
        if pred_weights.sum() == 0:
            pred_weights = np.ones_like(pred_weights) / len(pred_weights)
        else:
            pred_weights = pred_weights / pred_weights.sum()

        # Placeholder portfolio calculations
        daily_returns_mean = np.array([0.0005, 0.0006, 0.0007, 0.0008, 0.0004])
        daily_returns_cov = np.diag([0.0001, 0.00012, 0.00015, 0.0002, 0.0001])

        annual_return = np.dot(daily_returns_mean, pred_weights) * 252
        annual_vol = np.sqrt(np.dot(pred_weights.T, np.dot(daily_returns_cov * 252, pred_weights)))
        sharpe_ratio = annual_return / annual_vol

        portfolio = {tickers[i]: round(pred_weights[i], 4) for i in range(len(tickers))}

        return render_template('result.html',
                               portfolio=portfolio,
                               annual_return=round(annual_return * 100, 2),
                               annual_vol=round(annual_vol * 100, 2),
                               sharpe_ratio=round(sharpe_ratio, 2))
    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        return f"Error: {str(e)}"

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
