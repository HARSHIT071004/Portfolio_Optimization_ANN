# Portfolio Optimization ANN Web App

**Live Demo:** [invest-harshit-genius.onrender.com](https://invest-harshit-genius.onrender.com/)

A production-ready web application that uses a trained Artificial Neural Network to predict optimal portfolio allocations for top US stocks. Users input expected daily returns and receive recommended allocation weights, projected annual return, volatility, and Sharpe ratio.

The ANN forward pass is implemented in pure NumPy — no deep learning framework required at inference time. This keeps the deployment lightweight (~10 MB), starts instantly, and runs on Render's free tier without memory issues.

---

## How It Works

The application follows a three-step workflow:

1. **Enter Expected Returns** — Input your anticipated daily return for each stock (AAPL, MSFT, AMZN, TSLA, SPY) as a decimal. For example, `0.002` represents +0.2%. Use the "Sample Values" button to pre-fill reasonable estimates.

2. **ANN Predicts Allocation** — The trained neural network processes your inputs through four dense layers (64 → 128 → 64 → 5 neurons) with ReLU activations and a final softmax layer. The output is a set of portfolio weights that sum to 100%.

3. **Review Your Portfolio** — The results page displays:
   - **Portfolio Allocation** — Bar chart showing the recommended weight for each stock
   - **Annual Return** — Projected yearly return based on the allocation
   - **Volatility** — A measure of portfolio risk (lower = more stable)
   - **Sharpe Ratio** — Risk-adjusted return (above 1.0 is good, above 2.0 is excellent)
   - **Portfolio Insight** — Plain-English analysis of the top allocation and Sharpe ratio with actionable guidance

### Model Architecture

The ANN consists of 4 dense layers with 17,285 total parameters:

| Layer | Type | Input | Output | Activation |
|-------|------|-------|--------|------------|
| 1 | Dense | 5 | 64 | ReLU |
| 2 | Dense | 64 | 128 | ReLU |
| 3 | Dense | 128 | 64 | ReLU |
| 4 | Dense | 64 | 5 | Softmax |

The model was trained on historical stock data using a Jupyter notebook (`notebooks/portfolio_optimization_ANN.ipynb`). At inference time, the forward pass runs entirely in NumPy — weights are loaded from a `.npz` file and no TensorFlow dependency is required.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Inference | NumPy (forward pass only) |
| Training | TensorFlow / Keras (Jupyter notebook) |
| Frontend | HTML5, CSS3 (glassmorphism design) |
| Deployment | Render.com (Free tier) |
| Server | Gunicorn |

---

## Project Structure

```
Portfolio_Optimization_ANN/
│
├── app.py                    # Flask application (routes, prediction logic)
├── predict_numpy.py          # Pure NumPy ANN forward pass
├── requirements.txt          # Python dependencies (Flask, NumPy, gunicorn)
├── Procfile                  # Gunicorn start command for Render
├── .gitignore
├── README.md
│
├── model/
│   ├── portfolio_weights.npz # Extracted ANN weights (used by app)
│   └── portfolio_model.h5    # Full Keras model (for reference / retraining)
│
├── templates/
│   ├── index.html            # Home page with input form
│   └── result.html           # Portfolio results page
│
├── static/
│   └── style.css             # Application stylesheet
│
└── notebooks/
    └── portfolio_optimization_ANN.ipynb  # Model training notebook
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/HARSHIT071004/Portfolio_Optimization_ANN.git
cd Portfolio_Optimization_ANN

# Install dependencies
pip install -r requirements.txt
```

Dependencies: `Flask`, `numpy`, `gunicorn` — that's it. The model weights are pre-extracted in `model/portfolio_weights.npz`.

---

## Running Locally

```bash
# Start the Flask development server
python app.py

# Or via Flask CLI
flask run --host=0.0.0.0 --port=5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Enter expected daily returns for each stock and click "Predict Portfolio".

---

## Deployment on Render (Free Tier)

1. Push the repository to GitHub
2. In Render Dashboard, create a new **Web Service**
3. Connect your GitHub repository
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** (auto-detected from `Procfile`)
5. Deploy

The `Procfile` starts Gunicorn with a single worker, 2 threads, and a 120-second timeout:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --max-requests 10
```

Because the app uses NumPy instead of TensorFlow, cold starts are near-instant and memory usage stays well within Render's 512 MB free tier limit.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with input form |
| `/predict` | POST | Submit expected returns, receive portfolio prediction |
| `/check_model` | GET | Verify model weights are loaded |

### POST /predict

**Request body:** Form data with keys `AAPL`, `MSFT`, `AMZN`, `TSLA`, `SPY` (each a decimal representing expected daily return).

**Response:** Rendered HTML page showing:
- Portfolio allocation bar chart
- Annual return (%), volatility (%), Sharpe ratio
- Plain-English portfolio insight

**Example:**

```
AAPL=0.0015&MSFT=0.0018&AMZN=0.0022&TSLA=0.0035&SPY=0.0008
```

---

## Usage Guide

### Input Format

- Enter expected **daily return** as a decimal
- `0.002` = +0.2% expected daily return
- `-0.01` = -1.0% expected daily return
- Typical range: `-0.02` (-2%) to `+0.02` (+2%)

### Interpreting Results

- **Allocation %** — The percentage of your portfolio the model recommends for each stock. Higher weight indicates higher expected contribution to risk-adjusted returns.
- **Annual Return** — The projected yearly return assuming 252 trading days. Calculated as `daily_return * 252`.
- **Volatility** — Annualized portfolio risk (standard deviation of returns). Lower values suggest more stable performance.
- **Sharpe Ratio** — Return per unit of risk. Above 1.0 is acceptable, above 2.0 is excellent. If below 1.0, consider adjusting your expected returns.

---

## License

MIT
