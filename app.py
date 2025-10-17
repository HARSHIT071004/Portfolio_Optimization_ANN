from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load trained ANN model
model_path = os.path.join('model', 'portfolio_model.h5')
model = load_model(model_path, compile=False)

tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        user_input = [float(request.form[ticker]) for ticker in tickers]
        user_input = np.array(user_input).reshape(1, -1)

        # Predict weights
        pred_weights = model.predict(user_input).flatten()
        pred_weights = pred_weights / pred_weights.sum()  # normalize

        # Calculate portfolio metrics
        # Example: Expected return and volatility using mean/cov from historical daily_returns
        # For simplicity, placeholder values; in production, use historical data
        daily_returns_mean = np.array([0.0005, 0.0006, 0.0007, 0.0008, 0.0004])
        daily_returns_cov = np.diag([0.0001, 0.00012, 0.00015, 0.0002, 0.0001])
        annual_return = np.dot(daily_returns_mean, pred_weights) * 252
        annual_vol = np.sqrt(np.dot(pred_weights.T, np.dot(daily_returns_cov * 252, pred_weights)))
        sharpe_ratio = annual_return / annual_vol

        portfolio = {tickers[i]: round(pred_weights[i], 4) for i in range(len(tickers))}

        return render_template('result.html', portfolio=portfolio,
                               annual_return=round(annual_return*100,2),
                               annual_vol=round(annual_vol*100,2),
                               sharpe_ratio=round(sharpe_ratio,2))
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
