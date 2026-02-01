import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Gradient Boosting Libraries
from xgboost import XGBRegressor
from catboost import CatBoostRegressor # type: ignore
from sklearn.preprocessing import StandardScaler
import yfinance as yf # type: ignore

import datetime
import sys
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s- %(message)s',
    handlers=[
        logging.FileHandler("Risk_Scoring.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_data(company_symbol, period='1mo', interval='1d'):
    try: 
        logging.info(f"Fetching data for company {company_symbol}")
        data = yf.download(company_symbol, period=period, interval=interval)
        if data.empty:
            logging.error(f"Error While downloading the data")
            return None
        data.columns = ['_'.join(col).strip() for col in data.columns.to_flat_index()]
        data.reset_index(inplace=True)
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        logging.info(f"Successfully Retrieved data for company {company_symbol} for {len(data)} records")
        if data is None or len(data) < 2:
            logging.error("Insufficient data for training.")
            return None
        return data
    except:
        logging.error(f"Error occured while retrieving the data for company {company_symbol}")
        return None


def train_ensemble(data):
    """Train multiple models and return ensemble results"""
    data = data.dropna(subset=['prev_open', 'prev_high', 'prev_low', 'prev_vol', 'Lag1', 'Lag2', 'ma_10', 'RSI', 'MACD', 'BB_middle', 'OBV', 'Pivot', 'Close'])
    
    if data.shape[0] < 2:
        logging.error("Insufficient data for training. Need at least 2 samples.")
        return None
    
    # Feature selection
    feature_cols = ['prev_open', 'prev_high', 'prev_low', 'prev_vol', 'Lag1', 'Lag2', 'ma_10', 'RSI', 'MACD', 'BB_middle', 'OBV', 'Pivot']
    x = data[feature_cols]
    y = data['Close']
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Train all models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'SVR': SVR(C=1.0, kernel='rbf'),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.1, verbose=0, random_state=42)
    }
    
    trained_models = {}
    model_scores = {}
    
    for model_name, model in models.items():
        try:
            logging.info(f"Training {model_name}")
            model.fit(x_scaled, y)
            
            # Calculate score (R²)
            score = model.score(x_scaled, y)
            model_scores[model_name] = score
            trained_models[model_name] = model
            
            logging.info(f"{model_name} trained with R² score: {score:.4f}")
        except Exception as e:
            logging.error(f"Failed to train {model_name}: {e}")
    
    # Get feature importance from tree-based models
    feature_importance = {}
    
    # Random Forest importance
    if 'Random Forest' in trained_models:
        rf_importance = trained_models['Random Forest'].feature_importances_
        for i, col in enumerate(feature_cols):
            feature_importance[col] = rf_importance[i] * 100
    
    # XGBoost importance
    elif 'XGBoost' in trained_models:
        xgb_importance = trained_models['XGBoost'].feature_importances_
        for i, col in enumerate(feature_cols):
            feature_importance[col] = xgb_importance[i] * 100
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'models': trained_models,
        'scaler': scaler,
        'feature_importance': feature_importance,
        'model_scores': model_scores,
        'feature_cols': feature_cols
    }


def calculate_risk_score(ensemble_results, next_data, historical_data, current_price):
    """Calculate risk score from ensemble predictions"""
    try:
        models = ensemble_results['models']
        scaler = ensemble_results['scaler']
        feature_importance = ensemble_results['feature_importance']
        
        # Scale the next day data
        next_data_scaled = scaler.transform(next_data)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(next_data_scaled)[0]
            predictions[model_name] = pred
        
        # Calculate ensemble prediction (mean)
        predicted_price = np.mean(list(predictions.values()))
        
        # Calculate risk metrics
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Risk score calculation
        # Higher risk if predicting downside movement
        if price_change_pct < 0:
            # Downside predicted - high risk
            risk_from_price = min(100, abs(price_change_pct) * 20)  # Scale to 0-100
        else:
            # Upside predicted - lower risk
            risk_from_price = max(0, 40 - (price_change_pct * 5))  # Lower risk for upside
        
        # Add volatility-based risk
        volatility = historical_data['Close'].pct_change().std() * 100
        risk_from_volatility = min(40, volatility * 200)  # Up to 40 points from volatility
        
        # Add technical indicator risk
        rsi = historical_data['RSI'].iloc[-1]
        rsi_risk = 0
        if rsi > 70:
            rsi_risk = (rsi - 70) * 1.5  # Overbought risk
        elif rsi < 30:
            rsi_risk = (30 - rsi) * 1.5  # Oversold risk
        
        # Combine risk components
        total_risk = (risk_from_price * 0.5 + risk_from_volatility * 0.3 + rsi_risk * 0.2)
        risk_score = min(100, max(0, total_risk))
        
        # Classify risk
        if risk_score < 30:
            risk_class = "LOW RISK"
        elif risk_score < 60:
            risk_class = "MEDIUM RISK"
        else:
            risk_class = "HIGH RISK"
        
        # Calculate model agreement (how similar predictions are)
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        agreement = max(0, 100 - (pred_std / pred_mean * 100 if pred_mean != 0 else 50))
        
        # Model voting
        model_votes = {}
        for model_name, pred in predictions.items():
            model_price_change = ((pred - current_price) / current_price) * 100
            
            # Calculate individual model risk
            if model_price_change < -2:
                model_votes[model_name] = "HIGH"
            elif model_price_change < 0:
                model_votes[model_name] = "MEDIUM"
            else:
                model_votes[model_name] = "LOW"
        
        # Risk contributions (feature importance mapped to risk)
        risk_contributions = {}
        for feature, importance in list(feature_importance.items())[:6]:
            risk_contributions[feature] = importance
        
        # Generate explanation
        explanation = []
        if price_change_pct < -2:
            explanation.append(f"Models predict {abs(price_change_pct):.2f}% downside movement")
        elif price_change_pct < 0:
            explanation.append(f"Models predict moderate {abs(price_change_pct):.2f}% downside")
        else:
            explanation.append(f"Models predict {price_change_pct:.2f}% upside potential")
        
        if rsi > 70:
            explanation.append(f"RSI overbought at {rsi:.1f} - bearish signal")
        elif rsi < 30:
            explanation.append(f"RSI oversold at {rsi:.1f} - potential reversal")
        
        if volatility > 3:
            explanation.append(f"High volatility detected ({volatility:.2f}%) - increased uncertainty")
        
        # Top risk factor
        top_factor = list(feature_importance.keys())[0]
        explanation.append(f"{top_factor} is the primary risk factor ({feature_importance[top_factor]:.1f}% contribution)")
        
        return {
            'risk_score': risk_score,
            'risk_class': risk_class,
            'confidence': agreement,
            'downside_prob': min(100, risk_score * 1.2) if price_change_pct < 0 else max(0, risk_score * 0.8),
            'expected_downside': abs(price_change_pct) if price_change_pct < 0 else price_change_pct * 0.3,
            'model_agreement': agreement,
            'predicted_price': predicted_price,
            'model_votes': model_votes,
            'risk_contributions': risk_contributions,
            'explanation': explanation,
            'all_predictions': predictions
        }
    
    except Exception as e:
        logging.error(f"Error calculating risk score: {e}")
        return None


def get_next_day_data(data):
    try:
        latest_data = data.iloc[-1].copy()
        next_day = pd.DataFrame(columns=['prev_open', 'prev_high', 'prev_low', 'prev_vol', 'Lag1', 'Lag2', 'ma_10', 'RSI', 'MACD', 'BB_middle', 'OBV', 'Pivot'])
        next_day.loc[0, 'Lag1'] = latest_data['Close']
        next_day.loc[0, 'Lag2'] = latest_data['Lag1']
        next_day.loc[0, 'prev_high'] = latest_data['High']
        next_day.loc[0, 'prev_low'] = latest_data['Low']
        next_day.loc[0, 'prev_open'] = latest_data['Open']
        next_day.loc[0, 'prev_vol'] = latest_data['Volume']
        next_day.loc[0, 'ma_10'] = data['Close'].tail(10).mean()
        next_day.loc[0, 'RSI'] = latest_data['RSI']
        next_day.loc[0, 'MACD'] = latest_data['MACD']
        next_day.loc[0, 'BB_middle'] = latest_data['BB_middle']
        next_day.loc[0, 'OBV'] = latest_data['OBV']
        next_day.loc[0, 'Pivot'] = latest_data['Pivot']
        return next_day
    except Exception as e:
        logging.warning(f"Failed to calculate features for prediction: {e}")
        return None


def plot_price_analysis(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price', color='#00FF41', linewidth=2)
    ax.set_title('Price Trend Analysis', fontsize=14, color='white')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price ($)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1E2D3D')
    fig.patch.set_facecolor('#0E1117')
    ax.legend()
    return fig


def plot_vol_analysis(data):
    import matplotlib.dates as mdates   # ← add this import at top of file if not already there
    
    fig, ax = plt.subplots(figsize=(12, 6))   # slightly wider is better for intraday
    
    # Create color list
    colors = ['#00FF41' if close >= open_ else '#FF4444' 
              for close, open_ in zip(data['Close'], data['Open'])]
    
    # ─── Important: control bar width for intraday ───────────────────────
    # For 1-minute data: width = 0.6–0.9 of a minute in days fraction
    bar_width = 0.9 / (24 * 60)           # ≈ 0.9 minute wide bars
    
    ax.bar(data['Date'], data['Volume'], 
           width=bar_width,               # ← this is the key fix
           color=colors, 
           alpha=0.85,
           edgecolor='none',
           label="Volume")
    
    # Better date formatting for intraday
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=15))
    
    # Rotate labels if still crowded
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title("Volume Trend Analysis (colored by price direction)", 
                 fontsize=14, color='white')
    ax.set_xlabel('Time', fontsize=12, color='white')
    ax.set_ylabel('Volume', fontsize=12, color='white')
    
    ax.tick_params(colors='white')
    ax.set_facecolor('#1E2D3D')
    fig.patch.set_facecolor('#0E1117')
    
    ax.grid(True, alpha=0.15, color='gray')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_close_vs_ma(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'ma_5' in data.columns:
        ax.plot(data['Date'], data['ma_5'], label='5-day MA', color='#FF6347', linewidth=1.5)
    if 'ma_10' in data.columns:
        ax.plot(data['Date'], data['ma_10'], label='10-day MA', color='#FFA500', linewidth=1.5)
    ax.plot(data['Date'], data['Close'], label='Close Price', alpha=0.7, color='#00FF41', linewidth=2)
    ax.set_title('Moving Averages', fontsize=14, color='white')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price ($)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1E2D3D')
    fig.patch.set_facecolor('#0E1117')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_rsi(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['RSI'], label="RSI", color='#00FF41', linewidth=2)
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax.set_title("RSI Momentum Indicator", fontsize=14, color='white')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('RSI', fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1E2D3D')
    fig.patch.set_facecolor('#0E1117')
    ax.legend()
    return fig


def get_current_price(stock_symbol):
    try:
        ticker = yf.Ticker(stock_symbol)
        current_data = ticker.history(period="1d", interval="1m")
        current_price = current_data['Close'].iloc[-1]
        return current_price
    except Exception as e:
        logging.error(f"Error fetching current price for {stock_symbol}: {e}")
        return None


from alpha_vantage.timeseries import TimeSeries  # type:ignore
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"


def fetch_current_price(company_symbol, api_key=API_KEY):
    try:
        logging.info(f"Fetching real-time data for company {company_symbol}")
        ts = TimeSeries(key=api_key, output_format="pandas")
        data, meta_data = ts.get_quote_endpoint(symbol=company_symbol)
        
        if data.empty:
            logging.error("Error while downloading the data.")
            return None
        
        data.reset_index(inplace=True)
        data = data.iloc[0]
        
        formatted_data = pd.DataFrame([{
            "Date": pd.Timestamp.now(),
            "Open": float(data['02. open']),
            "High": float(data['03. high']),
            "Low": float(data['04. low']),
            "Close": float(data['05. price']),
            "Volume": int(data['06. volume']),
        }])
        
        logging.info(f"Successfully retrieved real-time data for {company_symbol}")
        return formatted_data['Close'].iloc[0]
    except Exception as e:
        logging.error(f"Error occurred while retrieving the data for {company_symbol}: {str(e)}")
        return None