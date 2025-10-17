from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable logging (for Render logs debugging)
logging.basicConfig(level=logging.INFO)

# Load trained ANN model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_model.h5')
if os.path.exists(model_path):
    model = load_model(model_path, compile=False)
    logging.info(f"✅ Model loaded successfully from {model_path}")
else:
    model = None
    logging.error("❌ Model file not found!")

# Define tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not found ❌"

    try:
        logging.info("Form data received: %s", request.form)

        # Get user input
        user_input = []
        for ticker in tickers:
            val = request.form.get(ticker)
            if val is None or val.strip() == "":
                return f"Error: Missing input for {ticker}"
            user_input.append(float(val))

        user_input = np.array(user_input).reshape(1, -1)
        logging.info(f"User input array: {user_input}")

        # Predict portfolio weights
        pred_weights = model.predict(user_input).flatten()
        logging.info(f"Raw predicted weights: {pred_weights}")

        if pred_weights.sum() == 0:
            pred_weights = np.ones_like(pred_weights) / len(pred_weights)
        else:
            pred_weights = pred_weights / pred_weights.sum()

        # Placeholder daily returns and covariance matrix
        daily_returns_mean = np.array([0.0005, 0.0006, 0.0007, 0.0008, 0.0004])
        daily_returns_cov = np.diag([0.0001, 0.00012, 0.00015, 0.0002, 0.0001])

        annual_return = np.dot(daily_returns_mean, pred_weights) * 252
        annual_vol = np.sqrt(np.dot(pred_weights.T, np.dot(daily_returns_cov * 252, pred_weights)))
        sharpe_ratio = annual_return / annual_vol

        portfolio = {tickers[i]: round(pred_weights[i], 4) for i in range(len(tickers))}

        logging.info(f"Final Portfolio: {portfolio}")
        logging.info(f"Metrics → Return: {annual_return}, Vol: {annual_vol}, Sharpe: {sharpe_ratio}")

        return render_template(
            'result.html',
            portfolio=portfolio,
            annual_return=round(annual_return * 100, 2),
            annual_vol=round(annual_vol * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 2)
        )

    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        return f"Error: {str(e)}"


@app.route('/check_model')
def check_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'portfolio_model.h5')
    if os.path.exists(model_path):
        return "Model exists ✅"
    else:
        return "Model missing ❌"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
