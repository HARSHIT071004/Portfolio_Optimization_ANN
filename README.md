📊 Portfolio Optimization ANN Web App

Live Demo: https://invest-harshit-genius.onrender.com

A Flask web application that predicts optimal portfolio allocations for a set of stocks using a trained Artificial Neural Network (ANN) model. Users can input expected returns for select stocks and get recommended weights, along with estimated annual return, volatility, and Sharpe ratio.

Features

Predict optimal portfolio weights for stocks: AAPL, MSFT, AMZN, TSLA, SPY

Lightweight ANN model integrated with Flask

Annualized portfolio return, volatility, and Sharpe ratio calculation

Minimal resource usage for Render free tier deployment

Logs user requests for easier debugging

Tech Stack

Backend: Python, Flask

Model: TensorFlow / Keras ANN

Frontend: HTML + CSS

Deployment: Render.com (Free tier)

Folder Structure
portfolio-flask-app/
│
├─ app.py                 # Flask application
├─ requirements.txt       # Python dependencies
├─ Procfile               # Gunicorn command for Render
├─ /model/
│   └─ portfolio_model.h5 # Trained ANN model
├─ /templates/
│   ├─ index.html         # Form page for user input
│   └─ result.html        # Display portfolio results
└─ /static/
    └─ style.css          # Optional styling

Installation

Clone the repo:

git clone <repo-url>
cd portfolio-flask-app


Install dependencies:

pip install -r requirements.txt


Ensure model exists:
Place portfolio_model.h5 inside the /model/ folder.

Running Locally
export FLASK_APP=app.py
export FLASK_ENV=development
flask run


Open http://127.0.0.1:5000
 in your browser.

Deployment on Render (Free Tier)

Push your repo to GitHub.

Create a new Web Service on Render.com
.

Connect your GitHub repo.

Build Command:

pip install -r requirements.txt


Start Command (Procfile):

web: gunicorn app:app --workers=1 --timeout 120


Render will automatically detect PORT environment variable.

Access your live app at https://<your-app-name>.onrender.com.

Usage

Navigate to /

Enter expected daily returns for each stock

Click Predict

View recommended portfolio weights and metrics

API Endpoints
Endpoint	Method	Description
/	GET	Home page with input form
/predict	POST	Predict portfolio weights
/check_model	GET	Check if model is loaded correctly
Notes / Optimizations

TensorFlow is optimized for low memory usage on free Render tier:

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


Single worker used to avoid SIGKILL / OOM errors

Logging enabled for request tracking:

logging.info("Form data received: %s", request.form)
logging.info("User input array: %s", user_input)

License

MIT License
