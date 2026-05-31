# Portfolio Optimization ANN — Complete Project & Model Training Flow

## 1. What Is This Project?

A **Flask web app** that predicts **optimal portfolio allocations** for 5 stocks (AAPL, MSFT, AMZN, TSLA, SPY) using an **Artificial Neural Network (ANN)** trained to learn the mapping from daily stock returns to optimal portfolio weights derived via **mean-variance optimization** (maximizing Sharpe ratio).

**Live Demo:** https://invest-harshit-genius.onrender.com/  
**Deployment:** Render.com free tier (single worker, no GPU, low memory)

---

## 2. Project Structure

```
D:\Portfolio_Optimization_ANN\
│
├── .gitignore
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── app.py                         # Flask web app (serves the model)
├── MODEL_TRAINING_FLOW.md         # This file
│
├── model\
│   └── portfolio_model.h5         # Trained ANN (~244 KB)
│
├── notebooks\
│   └── portfolio_optimization_ANN.ipynb  # Jupyter notebook — THE TRAINING PIPELINE
│
├── static\
│   └── style.css                  # Dark-themed glassmorphism CSS
│
└── templates\
    ├── index.html                 # Input form (enter expected daily returns)
    └── result.html                # Shows predicted weights + portfolio metrics
```

---

## 3. Libraries & Dependencies

| Library | Version | Purpose |
|---|---|---|
| Flask | 3.1.2 | Web framework |
| TensorFlow | 2.20.0 | Deep learning (training + inference) |
| Keras | 3.11.3 | High-level NN API |
| NumPy | latest | Numerical arrays |
| Pandas | latest | Data manipulation |
| scikit-learn | latest | `MinMaxScaler`, `train_test_split` |
| Matplotlib | latest | Plotting |
| Seaborn | latest | Statistical visualizations |
| yfinance | latest | Yahoo Finance data download |
| SciPy | (implicit) | `scipy.optimize.minimize` (SLSQP) |
| Gunicorn | latest | WSGI server for deployment |

---

## 4. Model Training Flow — Full Step-by-Step

Everything below happens inside **`notebooks/portfolio_optimization_ANN.ipynb`**.

### Step 1: Import Libraries
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 1

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize
```

---

### Step 2: Download Historical Stock Data
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 2

```python
tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'SPY']
data = yf.download(tickers, start="2018-01-01", end="2025-01-01")
adj_close = data['Close'].fillna(method='ffill')
```

- **Source:** Yahoo Finance via `yfinance`
- **Date range:** 2018-01-01 to 2025-01-01
- **Data:** Adjusted Close prices
- **Missing values:** Forward-filled
- **Output shape:** ~1761 rows × 5 columns

---

### Step 3: Calculate Daily Returns
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 4

```python
daily_returns = adj_close.pct_change().dropna()
```

- Converts prices to daily percentage returns
- Drops the first row (NaN from pct_change)
- **Output shape:** ~1760 × 5
- Visualizes correlation heatmap with seaborn

---

### Step 4: Generate Target Weights via Mean-Variance Optimization
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cells 5-6

This is the **core innovation** — generating the "ground truth" labels for the ANN.

#### Functions defined (Cell 5):

```python
def portfolio_metrics(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
    sharpe = ret / vol
    return ret, vol, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_metrics(weights, mean_returns, cov_matrix)[2]
```

#### Optimization setup:

```python
mean_returns = daily_returns.mean()     # Global mean (entire period)
cov_matrix = daily_returns.cov()        # Global covariance (entire period)
num_stocks = len(tickers)
bounds = tuple((0, 1) for _ in range(num_stocks))   # Long-only
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x)-1}  # Sum to 1
```

#### Target generation loop:

```python
y_train = []
X_train = daily_returns.values[1:]  # Features
for _ in range(len(X_train)):
    result = minimize(neg_sharpe, np.random.random(num_stocks),
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds,
                      constraints=constraints)
    y_train.append(result.x)
y_train = np.array(y_train)
```

**Key observation:** The optimization uses the **same global** `mean_returns` and `cov_matrix` for every sample. Each iteration starts with a different random initial guess, so SLSQP converges to slightly different (but nearly identical) optimal weight vectors. The targets don't actually vary meaningfully with the input returns.

#### Assign X and y (Cell 6):

```python
X = daily_returns.values[1:]    # Features: ~1760 × 5
y = np.array(y_train)           # Targets: ~1760 × 5
```

---

### Step 5: Train/Test Split & Scaling
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 7

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- **80/20 split** (random, not time-series aware)
- **MinMaxScaler** scales features to [0, 1]
- **Note:** The scaler is **NOT saved** for inference in `app.py`

---

### Step 6: Build & Train the ANN
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 8

#### Architecture:

```
┌──────────────────────────────────────┐
│ Input:  5 features (daily returns)   │
│                                      │
│ Dense(64, relu)        384 params    │
│ Dense(128, relu)       8,320 params  │
│ Dense(64, relu)        8,256 params  │
│ Dense(5, softmax)      325 params    │
│                                      │
│ Total: 17,285 trainable parameters   │
│ Optimizer: Adam                      │
│ Loss: Mean Squared Error (MSE)       │
└──────────────────────────────────────┘
```

#### Training:

```python
model = Sequential([
    Dense(64, input_dim=num_stocks, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_stocks, activation='softmax')  # weights sum to 1
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train,
                    epochs=50, batch_size=32, validation_split=0.2)
```

**Training stats:**
- **Epochs:** 50
- **Batch size:** 32
- **Validation split:** 20% of training data (~282 samples val, ~1,126 train)
- **Final training loss:** ~8.28e-7
- **Final validation loss:** ~2.52e-7
- **No overfitting** — both losses decrease together

---

### Step 7: Save Model
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 9

```python
model.save('model/portfolio_model.h5')
```

- Saved to `model/portfolio_model.h5` (legacy HDF5 format)
- Warning about `.keras` format being preferred

---

### Step 8: Evaluate Model
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 10

```python
loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)
```

- **Test loss:** ~1.86e-7 (extremely low)
- Plots training vs validation loss curves

---

### Step 9: Make Predictions
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cell 11

```python
pred_weights = model.predict(X_test_scaled[-1].reshape(1,-1)).flatten()
pred_weights = pred_weights / pred_weights.sum()
```

**Example output:**

| Stock | Predicted Weight |
|---|---|
| AAPL | 0.3997 |
| MSFT | 0.0002 |
| AMZN | 0.3446 |
| TSLA | 0.0002 |
| SPY | 0.2553 |

---

### Step 10: Portfolio Performance
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cells 12-13

```python
initial_investment = 100000
portfolio_values = (daily_returns @ pred_weights + 1).cumprod() * initial_investment

annual_return = np.dot(daily_returns.mean(), pred_weights) * 252
annual_vol = np.sqrt(np.dot(pred_weights.T, np.dot(daily_returns.cov()*252, pred_weights)))
sharpe_ratio = annual_return / annual_vol
```

**Results:**
| Metric | Value |
|---|---|
| Expected Annual Return | 37.87% |
| Portfolio Volatility | 31.73% |
| Sharpe Ratio | 1.19 |

---

### Step 11: Visualizations
**File:** `notebooks/portfolio_optimization_ANN.ipynb` — Cells 14-18

1. **Bar chart** — Predicted portfolio weights
2. **Bar chart** — Annualized volatility per stock
3. **Line chart** — Portfolio growth over time ($100K → ~$2M)
4. **Heatmap** — Stock correlation matrix
5. **Scatter plot** — Risk vs predicted weight (annotated per stock)

---

## 5. Inference Flow — `app.py`

When a user submits expected daily returns via the web form:

```
User Browser                    Flask Server                   ANN Model
     │                              │                            │
     │   POST /predict              │                            │
     │   {AAPL:0.001, MSFT:...}     │                            │
     │─────────────────────────────►│                            │
     │                              │   Reshape → (1,5) array    │
     │                              │───────────────────────────►│
     │                              │   model.predict()          │
     │                              │◄───────────────────────────│
     │                              │                            │
     │                              │   Normalize weights (sum=1)│
     │                              │   Compute metrics using    │
     │                              │   hardcoded placeholder    │
     │                              │   mean/cov arrays          │
     │                              │                            │
     │   result.html                │                            │
     │   {AAPL:0.32, MSFT:0.18,    │                            │
     │    Return:15.2%, Vol:22.5%, │                            │
     │    Sharpe:0.68}              │                            │
     │◄─────────────────────────────│                            │
```

**Key issues in `app.py`:**
1. **No scaler** — The `MinMaxScaler` fitted during training is never saved or applied during inference; user raw input goes directly to the model
2. **Hardcoded metrics** — `daily_returns_mean` and `daily_returns_cov` in `app.py` are fixed placeholders, not derived from actual data
3. **TensorFlow optimizations** for low-memory Render deployment:
   - `tf.config.set_visible_devices([], 'GPU')` — disables GPU
   - Thread parallelism set to 1

---

## 6. Data Pipeline Summary

```
yfinance.download(tickers, 2018-01-01 → 2025-01-01)
         │
         ▼
adj_close = data['Close'].fillna(method='ffill')     [~1761 × 5]
         │
         ▼
daily_returns = adj_close.pct_change().dropna()      [~1760 × 5]
         │
         ├──► X = daily_returns.values[1:]           [~1760 × 5]
         │
         │    For each row (1760 iterations):
         │      minimize(neg_sharpe, random_init,
         │               args=(global_mean, global_cov),
         │               method='SLSQP', bounds, constraints)
         │
         └──► y = np.array(y_train)                  [~1760 × 5]
                  │
                  ▼
         train_test_split(X, y, test_size=0.2)
                  │
         ┌────────┴────────┐
         ▼                  ▼
    X_train (~1408)     X_test (~352)
    y_train (~1408)     y_test (~352)
         │
         ▼
    MinMaxScaler
         │
         ▼
    X_train_scaled, X_test_scaled
         │
         ▼
    model.fit(epochs=50, batch_size=32, val_split=0.2)
         │
         ├── Train loss → ~8.3e-7
         ├── Val loss   → ~2.5e-7
         ├── Test loss  → ~1.9e-7
         │
         ▼
    model.save('model/portfolio_model.h5')
```

---

## 7. Model Architecture Diagram

```
Input Layer               Hidden Layers              Output Layer
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐
│ AAPL     │    │          │    │           │    │          │    │ AAPL     │
│ Return   │───►│ Dense(64)│───►│ Dense(128)│───►│ Dense(64)│───►│ Weight   │
├─────────┤    │  ReLU    │    │  ReLU     │    │  ReLU    │    ├─────────┤
│ MSFT     │    │          │    │           │    │          │    │ MSFT     │
│ Return   │───►│          │───►│           │───►│          │───►│ Weight   │
├─────────┤    │          │    │           │    │          │    ├─────────┤
│ AMZN     │───►│          │───►│           │───►│          │───►│ AMZN     │
│ Return   │    │          │    │           │    │          │    │ Weight   │
├─────────┤    │          │    │           │    │          │    ├─────────┤
│ TSLA     │───►│          │───►│           │───►│          │───►│ TSLA     │
│ Return   │    │          │    │           │    │          │    │ Weight   │
├─────────┤    │          │    │           │    │          │    ├─────────┤
│ SPY      │───►│          │───►│           │───►│          │───►│ SPY      │
│ Return   │    │          │    │           │    │          │    │ Weight   │
└─────────┘    └──────────┘    └───────────┘    └──────────┘    └─────────┘
                                                                    │
                                                         Softmax (sum = 1)
```

---

## 8. Known Issues & Observations

| Issue | Details |
|---|---|
| **Target generation** | Uses global mean/cov for all samples — targets don't vary with input |
| **Missing scaler** | `MinMaxScaler` not saved/used in `app.py` — raw input goes to model |
| **Hardcoded metrics** | Portfolio return/vol/SR in app.py use fixed placeholder arrays |
| **No time-series split** | Random split may leak future info into training |
| **Legacy format** | Model saved as `.h5` (old), recommended `.keras` |
| **Limited variability** | With softmax + long-only constraints, model tends to concentrate on few stocks |
