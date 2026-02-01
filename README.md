# 🎯 Probabilistic Multi-Model Market Risk Predictor

**AI-Powered Stock Downside Risk Scoring System**  
Built for PS-17 Hackathon | Team BumbleTech

## 🚀 Overview

This is an **interactive Streamlit web application** that helps traders and investors quickly assess **downside risk** in stocks using:

- Real-time & historical market data (via yfinance)
- 12+ technical indicators & risk factors
- Ensemble of **8 classic machine learning models**
- Expert rule-based interpretable risk engine
- Beautiful dark-themed visualizations & explanations

The app outputs a **0–100 Risk Score**, risk class (LOW / MEDIUM / HIGH), model consensus voting, feature importance, and plain-English explanations — perfect for intraday, swing, and positional trading.

**Key PS-17 highlights covered:**
- Structured data processing
- Feature engineering (technical indicators)
- Multiple ML models + ensemble
- Model interpretability & voting
- Visualization + explainability
- Real-world finance use-case

## ✨ Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| Multi-timeframe analysis         | Short-term (1–30 min), Mid-term (1–2 hr), Long-term (days/weeks)           |
| 8-model ensemble                 | Linear, Ridge, Lasso, RandomForest, GBM, SVR, XGBoost, CatBoost            |
| Technical indicators             | RSI, MACD, ATR, Bollinger Bands, OBV, MFI, Pivot points, Moving Averages   |
| Expert rule-based scoring        | Human-interpretable rules (overbought, divergence, volume selling, etc.)   |
| Risk Gauge + Projection Chart    | Interactive Plotly gauge + risk-adjusted price path visualization          |
| Model Voting & Explanations      | See which models say HIGH/MEDIUM/LOW + why the score came out this way     |
| Dark cyberpunk UI                | Modern, eye-friendly theme with glowing accents                             |
| Data export                      | Download full feature-engineered dataset as CSV                            |

## 🛠️ Tech Stack

| Category             | Technologies / Libraries                                      |
|----------------------|----------------------------------------------------------------|
| Frontend             | Streamlit, Custom CSS, Plotly, Matplotlib                      |
| Data & Finance       | yfinance, pandas, numpy, ta-lib (technical analysis)           |
| Machine Learning     | scikit-learn, xgboost, catboost                                |
| Visualization        | Plotly (gauges, candlesticks), Matplotlib (classic charts)     |
| Logging & Utils      | Python logging, datetime                                       |

## 📂 Project Structure


market-risk-predictor/
├── app2.py                 # Main Streamlit application
├── model.py                # Data fetching, model training, risk scoring logic
├── tech.py                 # Enhanced indicators + expert rule system + gauge
├── pattern_scanner.py      # Chart pattern detection (double top, H&S, S/R)
├── styles.css              # Custom dark theme styling
├── Risk_Scoring.log        # Application logs (generated)
└── README.md
text


## 🚀 Quick Start (Local Setup)

### Prerequisites

- Python 3.9+
- pip
- Git (optional)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/market-risk-predictor.git
cd market-risk-predictor

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
yfinance
ta
plotly

Run the app
Bashstreamlit run app2.py
→ Open http://localhost:8501 in your browser
🔧 How to Use

Enter stock symbol (e.g. AAPL, RELIANCE.NS, BTC-USD)
Choose time horizon (Short / Mid / Long Term)
Select risk assessment duration
Click Analyze
Explore tabs:
📊 Risk Dashboard
📈 Risk Factor Analysis
🎯 Risk Assessment (main ML + expert scoring)
📑 Data & Metrics


⚠️ Important Notes

Not financial advice — For educational & research purpose only
Some features (real-time quotes) may require Alpha Vantage API key (currently commented)
Model training happens on-the-fly → needs at least ~50–100 candles for decent results
Intraday data limited by yfinance (sometimes 7 days max for 1-min data)


