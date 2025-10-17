📊 Portfolio Optimization ANN Web App

Live Demo: [View App](https://invest-harshit-genius.onrender.com/)

A Flask web app that predicts optimal portfolio allocations for selected stocks using a trained ANN model. Users input expected daily returns and get recommended weights, along with annualized return, volatility, and Sharpe ratio. Optimized for low-resource deployment on Render free tier.

✨ Features

Predict portfolio weights for AAPL, MSFT, AMZN, TSLA, SPY

Lightweight ANN model integrated with Flask

Calculate annual return, volatility, and Sharpe ratio

Minimal resource usage for Render free tier

Logging enabled for request tracking

🛠 Tech Stack

Backend: Python, Flask

Model: TensorFlow / Keras ANN

Frontend: HTML + CSS

Deployment: Render.com (Free tier)

📁 Folder Structure
portfolio-flask-app/
│
├─ app.py                 # Flask application
├─ requirements.txt       # Python dependencies
├─ Procfile               # Gunicorn command for Render
├─ /model/
│   └─ portfolio_model.h5 # Trained ANN model
├─ /templates/
│   ├─ index.html         # Form page
│   └─ result.html        # Portfolio results
└─ /static/
    └─ style.css          # Optional styling

⚡ Installation
git clone <repo-url>
cd portfolio-flask-app
pip install -r requirements.txt


Ensure: portfolio_model.h5 is inside /model/

🚀 Running Locally
export FLASK_APP=app.py
export FLASK_ENV=development
flask run


Open http://127.0.0.1:5000 in your browser.

🌐 Deployment on Render (Free Tier)

Push repo to GitHub

Create Web Service on Render

Connect GitHub repo

Build command: pip install -r requirements.txt

Start command (Procfile):

web: gunicorn app:app --workers=1 --timeout 120


Access live app at Render URL

🔗 API Endpoints
Endpoint	Method	Description
/	GET	Home page with input form
/predict	POST	Predict portfolio weights
/check_model	GET	Check if model is loaded
📝 Notes / Optimizations

TensorFlow optimized for low memory:

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


Single worker to prevent OOM / SIGKILL errors

Logging enabled:

logging.info("Form data received: %s", request.form)
logging.info("User input array: %s", user_input)

📜 License

MIT License
