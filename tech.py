import logging
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import ta  # type: ignore
import plotly.graph_objects as go  # type: ignore

def cal_enhanced_features(data):
    """Calculate additional technical indicators useful for intraday trading"""
    try:
        # Existing features
        data['Lag1'] = data['Close'].shift(1)
        data['Lag2'] = data['Close'].shift(2)
        data['prev_high'] = data['High'].shift(1)
        data['prev_low'] = data['Low'].shift(1)
        data['prev_open'] = data['Open'].shift(1)
        data['prev_vol'] = data['Volume'].shift(1)
        
        # Moving Averages
        data['ma_5'] = data['Close'].rolling(5).mean()
        data['ma_10'] = data['Close'].rolling(10).mean()
        data['ma_20'] = data['Close'].rolling(20).mean()
        data['ma_50'] = data['Close'].rolling(50).mean()
        data['ema_9'] = ta.trend.ema_indicator(data['Close'], 9)
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Volatility Indicators
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_middle'] = bb.bollinger_mavg()
        data['BB_lower'] = bb.bollinger_lband()        
        
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        
        # Volume Indicators
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        # Support and Resistance
        data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['R1'] = 2 * data['Pivot'] - data['Low']
        data['S1'] = 2 * data['Pivot'] - data['High']
        
        return data

    except Exception as e:
        print(f"Error in calc features: {str(e)}")
        raise
    

def get_intraday_signals(data):
    """Generate trading signals based on technical indicators"""
    signals = pd.DataFrame(index=data.index)
    
    # MACD Signal
    signals['MACD_Signal'] = np.where(data['MACD'] > 0, 1, -1)
    
    # RSI Signals
    signals['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    
    # Bollinger Bands Signals
    signals['BB_Signal'] = np.where(data['Close'] < data['BB_lower'], 1, 
                                  np.where(data['Close'] > data['BB_upper'], -1, 0))
    
    # Volume Signal
    signals['Volume_Signal'] = np.where(data['Volume'] > data['Volume_SMA'] * 1.5, 1, 0)
    
    # Combined Signal
    signals['Combined_Signal'] = (signals['MACD_Signal'] + signals['RSI_Signal'] + 
                                signals['BB_Signal'] + signals['Volume_Signal'])
    
    return signals


def plot_advanced_charts(data):
    """Generate advanced technical analysis charts"""
    figures = {}
    
    # Price and Volume Chart
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data['Date'], data['Close'], label='Close Price', color='#00FF88', linewidth=2)
    ax1.plot(data['Date'], data['BB_upper'], 'r--', label='BB Upper', linewidth=1.5)
    ax1.plot(data['Date'], data['BB_lower'], 'r--', label='BB Lower', linewidth=1.5)
    ax1.fill_between(data['Date'], data['BB_upper'], data['BB_lower'], alpha=0.1, color='red')
    ax1.set_title('Price with Bollinger Bands', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(data['Date'], data['Volume'], label='Volume', color='#00AAFF', alpha=0.7)
    ax2.plot(data['Date'], data['Volume_SMA'], 'r', label='Volume SMA', linewidth=2)
    ax2.set_title('Volume Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures['price_volume'] = fig1
    
    # Technical Indicators Chart
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    ax1.plot(data['Date'], data['RSI'], label='RSI', color='purple', linewidth=2)
    ax1.axhline(y=70, color='r', linestyle='--', label='Overbought')
    ax1.axhline(y=30, color='g', linestyle='--', label='Oversold')
    ax1.fill_between(data['Date'], 30, 70, alpha=0.1, color='gray')
    ax1.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RSI', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(data['Date'], data['MACD'], label='MACD', color='blue', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.fill_between(data['Date'], 0, data['MACD'], where=(data['MACD'] > 0), 
                     color='green', alpha=0.3, label='Bullish')
    ax2.fill_between(data['Date'], 0, data['MACD'], where=(data['MACD'] < 0), 
                     color='red', alpha=0.3, label='Bearish')
    ax2.set_title('MACD (Moving Average Convergence Divergence)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(data['Date'], data['MFI'], label='MFI', color='orange', linewidth=2)
    ax3.axhline(y=80, color='r', linestyle='--', label='Overbought')
    ax3.axhline(y=20, color='g', linestyle='--', label='Oversold')
    ax3.fill_between(data['Date'], 20, 80, alpha=0.1, color='gray')
    ax3.set_title('Money Flow Index (MFI)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('MFI', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures['indicators'] = fig2
    
    return figures


# NEW FUNCTION: Expert Rule-Based System
def expert_risk_system(data):
    """
    Human expert-level rule-based risk assessment
    This is a critical PS-17 feature demonstrating interpretable decision-making
    
    Returns:
        risk_score: 0-100 risk percentage
        risk_class: LOW/MEDIUM/HIGH
        risk_factors: List of textual explanations
    """
    if data.empty:
        return 50, "MEDIUM", ["Insufficient data for expert analysis"]
    
    latest = data.iloc[-1]
    risk_score = 0
    risk_factors = []
    
    # Rule 1: RSI Overbought/Oversold Analysis
    if latest['RSI'] > 70:
        contribution = min(25, (latest['RSI'] - 70) * 1.25)
        risk_score += contribution
        risk_factors.append(f"⚠️ **RSI Overbought** - RSI at {latest['RSI']:.1f} (>70) indicates potential downward correction. Contribution: +{contribution:.0f} points")
    elif latest['RSI'] < 30:
        contribution = 15
        risk_score -= contribution
        risk_factors.append(f"✅ **RSI Oversold** - RSI at {latest['RSI']:.1f} (<30) suggests limited downside risk. Contribution: -{contribution:.0f} points")
    else:
        risk_factors.append(f"ℹ️ **RSI Neutral** - RSI at {latest['RSI']:.1f} in normal range (30-70)")
    
    # Rule 2: MACD Momentum Analysis
    if latest['MACD'] < 0 and latest['Close'] > data['ma_20'].iloc[-1]:
        contribution = 20
        risk_score += contribution
        risk_factors.append(f"⚠️ **Bearish MACD Divergence** - Negative MACD ({latest['MACD']:.2f}) while price above 20-MA suggests momentum weakness. Contribution: +{contribution:.0f} points")
    elif latest['MACD'] > 0:
        risk_factors.append(f"✅ **Bullish MACD** - Positive momentum signal ({latest['MACD']:.2f})")
    else:
        risk_factors.append(f"⚠️ **Bearish MACD** - Negative momentum ({latest['MACD']:.2f})")
    
    # Rule 3: Volume Analysis
    if 'Volume_SMA' in data.columns and pd.notna(latest['Volume_SMA']):
        if latest['Volume'] > latest['Volume_SMA'] * 1.8:
            if latest['Close'] < latest['Open']:
                contribution = 30
                risk_score += contribution
                volume_ratio = latest['Volume'] / latest['Volume_SMA']
                risk_factors.append(f"⚠️ **High Volume Selling** - Volume {volume_ratio:.1f}x above average with price decline indicates strong selling pressure. Contribution: +{contribution:.0f} points")
            else:
                risk_factors.append(f"✅ **High Volume Buying** - Strong volume with price increase is bullish")
        else:
            risk_factors.append(f"ℹ️ **Normal Volume** - Volume within typical range")
    
    # Rule 4: Bollinger Bands Analysis
    if latest['Close'] > latest['BB_upper']:
        contribution = 15
        risk_score += contribution
        distance = ((latest['Close'] - latest['BB_upper']) / latest['BB_upper']) * 100
        risk_factors.append(f"⚠️ **Price Overextended** - Trading {distance:.1f}% above upper Bollinger Band suggests overbought condition. Contribution: +{contribution:.0f} points")
    elif latest['Close'] < latest['BB_lower']:
        contribution = 10
        risk_score -= contribution
        distance = ((latest['BB_lower'] - latest['Close']) / latest['BB_lower']) * 100
        risk_factors.append(f"✅ **Price Undervalued** - Trading {distance:.1f}% below lower Bollinger Band suggests potential reversal. Contribution: -{contribution:.0f} points")
    else:
        risk_factors.append(f"ℹ️ **Price in Normal Range** - Within Bollinger Bands")
    
    # Rule 5: Support/Resistance Proximity
    if latest['Close'] > latest['R1'] * 0.98:
        contribution = 20
        risk_score += contribution
        risk_factors.append(f"⚠️ **Near Resistance** - Price at ${latest['Close']:.2f} approaching resistance at ${latest['R1']:.2f}. High probability of rejection. Contribution: +{contribution:.0f} points")
    elif latest['Close'] < latest['S1'] * 1.02:
        risk_factors.append(f"✅ **Near Support** - Price at ${latest['Close']:.2f} near support at ${latest['S1']:.2f} provides downside protection")
    
    # Rule 6: ATR Volatility Analysis
    if 'ATR' in data.columns and pd.notna(latest['ATR']):
        atr_pct = (latest['ATR'] / latest['Close']) * 100
        if atr_pct > 3:  # High volatility
            contribution = 10
            risk_score += contribution
            risk_factors.append(f"⚠️ **High Volatility** - ATR indicates {atr_pct:.1f}% daily range, increasing uncertainty. Contribution: +{contribution:.0f} points")
        else:
            risk_factors.append(f"ℹ️ **Normal Volatility** - ATR at {atr_pct:.1f}% of price")
    
    # Rule 7: Moving Average Trend
    if 'ma_50' in data.columns and pd.notna(latest['ma_50']):
        if latest['Close'] < latest['ma_50'] and latest['ma_10'] < latest['ma_50']:
            contribution = 15
            risk_score += contribution
            risk_factors.append(f"⚠️ **Downtrend Confirmed** - Price and short-term MA below long-term MA. Contribution: +{contribution:.0f} points")
        elif latest['Close'] > latest['ma_50'] and latest['ma_10'] > latest['ma_50']:
            risk_factors.append(f"✅ **Uptrend Confirmed** - Price and short-term MA above long-term MA")
    
    # Normalize risk score to 0-100 range
    risk_score = max(0, min(100, risk_score + 50))  # Baseline at 50, adjust from there
    
    # Classify risk
    if risk_score > 60:
        risk_class = "HIGH"
    elif risk_score > 30:
        risk_class = "MEDIUM"
    else:
        risk_class = "LOW"
    
    return risk_score, risk_class, risk_factors


# NEW FUNCTION: Risk Gauge Visualization
def plot_risk_gauge(risk_score, risk_class):
    """
    Create an interactive gauge chart for risk visualization
    """
    # Determine color based on risk level
    if risk_class == "HIGH":
        bar_color = "#FF4444"
    elif risk_class == "MEDIUM":
        bar_color = "#FFAA00"
    else:
        bar_color = "#00FF88"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Downside Risk Score", 'font': {'size': 24, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "#00FF41",
            'steps': [
                {'range': [0, 30], 'color': "#003320"},
                {'range': [30, 60], 'color': "#332200"},
                {'range': [60, 100], 'color': "#330000"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={'color': "white", 'family': "Arial"},
        height=400
    )
    
    return fig


def footer_set():
    st.markdown("""
    <style>
        .footer {
            position: relative;
            top: 200px;
            bottom: 0;
            width: 100vw;
            left: -23vw;
            background-color: #000000;
            color: white;
            text-align: center;
            padding: 1rem;
            font-family: Arial, sans-serif;
            font-size: 0.9rem;
            box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.2);
        }
        .footer a {
            color: #ff6347;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class='footer'>
        <p style="padding-left: 250px;">⚠️ AI-Powered Market Risk Scoring System | PS-17 Hackathon Project</p>
        <p style="padding-left: 250px;">By <a href="https://www.instagram.com/sb_ritik" target="_blank">Team BumbleTech</a></p>
    </div>
""", unsafe_allow_html=True)