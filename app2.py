import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
import datetime
import streamlit as st
from model import get_data, train_ensemble, get_next_day_data, calculate_risk_score, plot_price_analysis, plot_vol_analysis, plot_close_vs_ma, plot_rsi, fetch_current_price, get_current_price
from tech import get_intraday_signals, cal_enhanced_features, plot_advanced_charts, footer_set
from pattern_scanner import implement_pattern_scanner_tab
import ta #type:ignore

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler("Risk_Scoring.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Page configuration
st.set_page_config(
    page_title="Market Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
<style>
    
    /* Main theme colors */
    :root {
        --background-color: #0E1117;
        --text-color: #E0E0E0;
        --accent-color: #00FF41;
        --secondary-color: #1E2D3D;
        --hover-color: #2C3E50;
        --risk-high: #FF4444;
        --risk-medium: #FFA500;
        --risk-low: #00FF41;
    }
    
    /* Global styles */
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Header styling */
    .main h1 {
        color: var(--accent-color);
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E2D3D 0%, #0E1117 100%);
        border: 1px solid var(--accent-color);
        border-radius: 5px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--secondary-color);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: var(--secondary-color);
        color: var(--accent-color);
        border: 1px solid var(--accent-color);
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--accent-color);
        color: var(--background-color);
    }
    
    /* Metric containers */
    [data-testid="stMetric"] {
        background-color: var(--secondary-color);
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid var(--accent-color);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--accent-color) !important;
    }
    
    /* Select boxes */
    .stSelectbox {
        background-color: var(--secondary-color);
    }
    
    /* DataFrames */
    .dataframe {
        background-color: var(--secondary-color);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: var(--accent-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--secondary-color);
        border-radius: 5px;
    }
    
    /* Card styling */
    .card {
        background-color: var(--secondary-color);
        border: 1px solid var(--accent-color);
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Risk gauge styling */
    .risk-gauge {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00FF41 0%, #00AA2B 100%);
        color: #000;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: #000;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
        color: #FFF;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("""
    <style>
        .glowing-title {
            background: linear-gradient(to right, #0a0a0a, #1b1b1b);
            color: #00ff00;
            font-weight: bold;
            font-size: 2.5rem;
            padding: 20px 0;
            text-align: center;
            border: 4px solid #00ff00;
            border-radius: 15px;
            box-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00;
            width: 100%;
            margin: 0;
        }
    </style>
    <h1 class="glowing-title">🎯 Probabilistic Multi-Model Market Risk Predictor</h1>
""", unsafe_allow_html=True)





# Session state initialization
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}
if 'risk_scores' not in st.session_state:
    st.session_state['risk_scores'] = {}

# Sidebar
with st.sidebar:
    st.markdown("### 🛠️ **Risk Analysis Configuration**")
    
    # Stock symbol input
    company_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", key='symbol')
    
    # Analysis term selection
    st.markdown("### Time Horizon")
    prediction_term = st.selectbox(
        'Select Analysis Period',
        ['📅 Short Term', '📅 Mid Term', '📅 Long Term']
    )
    
    # Time selectors
    if 'Short Term' in prediction_term:
        days_to_predict = st.selectbox(
            'Minutes to Assess Risk',
            options=[1, 2, 5, 15, 30]
        )
        analysis_period = st.selectbox(
            'Historical Data Period (Days)',
            options=[1, 5]
        )
        
    elif 'Mid Term' in prediction_term:
        days_to_predict = st.selectbox(
            'Minutes to Assess Risk',
            options=[60, 90]
        )
        analysis_period = st.selectbox(
            'Historical Data Period (Days)',
            options=[5, 30]
        )
        
    else:
        days_to_predict = st.selectbox(
            'Days to Assess Risk',
            options=[1, 5]
        )
        analysis_period = st.selectbox(
            'Historical Data Period (Months)',
            options=[1, 3, 6, 12]
        )
    
    st.markdown("---")
    
    # Risk threshold settings
    st.markdown("### ⚙️ Risk Thresholds")
    risk_threshold_low = st.slider("Low Risk Threshold", 0, 50, 30)
    risk_threshold_high = st.slider("High Risk Threshold", 50, 100, 60)
    
    st.markdown("---")
    
    # Info box
    st.info("""
    📊 **How it works:**
    
    1. Fetches market data
    2. Extracts 12+ risk factors
    3. Trains 8 ML models
    4. Calculates ensemble risk score
    5. Provides interpretable results
    """)

# Main content
if company_symbol:
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch data
        status_text.text("📡 Fetching market data...")
        progress_bar.progress(20)
        
        if 'Short' in prediction_term:
            period = f'{analysis_period}d'
            interval = '1m'
        elif 'Mid' in prediction_term:
            period = f'{analysis_period}d'
            interval = '5m'
        else:
            period = f'{analysis_period}mo'
            interval = '1d'
        
        data = get_data(company_symbol, period=period, interval=interval)
        
        if data is None or len(data) < 2:
            st.error("⚠️ Unable to fetch data. Please check the symbol and try again.")
            st.stop()
        
        st.session_state['datasets'][company_symbol] = data
        
        # Step 2: Calculate features
        status_text.text("🔧 Calculating risk factors...")
        progress_bar.progress(40)
        
        featured_data = cal_enhanced_features(data)
        
        # Get current price
        try:
            current_price = get_current_price(company_symbol)
            if current_price is None:
                current_price = data['Close'].iloc[-1]
        except:
            current_price = data['Close'].iloc[-1]
        
        # Calculate price change
        if len(data) > 1:
            price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        else:
            price_change = 0
        
        status_text.text("✅ Data prepared successfully!")
        progress_bar.progress(60)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logging.error(f"Data preparation error: {e}")
        st.stop()
    
    # Create tabs
    tabs = st.tabs([
        "📊 Risk Dashboard", 
        "📈 Risk Factor Analysis",
        "🎯 Risk Assessment",
        "📑 Data & Metrics"
    ])
    
    # Tab 1: Risk Dashboard
    with tabs[0]:
        st.markdown("## 📊 Market Risk Overview")
        
        # Current market status
        st.markdown("### 💹 Current Market Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:.2f}%"
            )
        
        with col2:
            volatility = featured_data['ATR'].iloc[-1] if 'ATR' in featured_data.columns else 0
            st.metric(
                "Volatility (ATR)",
                f"{volatility:.2f}",
                "Risk Factor"
            )
        
        with col3:
            rsi = featured_data['RSI'].iloc[-1] if 'RSI' in featured_data.columns else 50
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                "RSI",
                f"{rsi:.1f}",
                rsi_status
            )
        
        with col4:
            volume_ratio = featured_data['Volume'].iloc[-1] / featured_data['Volume'].mean() if 'Volume' in featured_data.columns else 1
            st.metric(
                "Volume Ratio",
                f"{volume_ratio:.2f}x",
                "vs Average"
            )
        
        # Price chart
        st.markdown("### 📈 Price Trend Analysis")
        fig = plot_price_analysis(data)
        st.pyplot(fig)
        
        # Volume analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Volume Trend")
            fig_vol = plot_vol_analysis(data)
            st.pyplot(fig_vol)
        
        with col2:
            st.markdown("### 📉 Moving Averages")
            fig_ma = plot_close_vs_ma(featured_data)
            st.pyplot(fig_ma)
        
        footer_set()
    
    # Tab 2: Risk Factor Analysis
    with tabs[1]:
        st.markdown("## 📈 Risk Factor Analysis")
        
        # Generate signals
        signals = get_intraday_signals(featured_data)
        
        # Risk factors overview
        st.markdown("### 🎯 Key Risk Indicators")
        
        # Create 3 columns for risk factor categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Momentum Risk")
            rsi_val = featured_data['RSI'].iloc[-1]
            rsi_risk = "HIGH" if rsi_val > 70 or rsi_val < 30 else "MEDIUM" if rsi_val > 60 or rsi_val < 40 else "LOW"
            st.metric("RSI Risk", rsi_risk, f"{rsi_val:.2f}")
            
            macd_val = featured_data['MACD'].iloc[-1]
            macd_risk = "Bearish" if macd_val < 0 else "Bullish"
            st.metric("MACD Signal", macd_risk, f"{macd_val:.4f}")
        
        with col2:
            st.markdown("#### Volatility Risk")
            atr_val = featured_data['ATR'].iloc[-1]
            atr_mean = featured_data['ATR'].mean()
            volatility_risk = "HIGH" if atr_val > atr_mean * 1.5 else "MEDIUM" if atr_val > atr_mean else "LOW"
            st.metric("ATR Risk", volatility_risk, f"{atr_val:.2f}")
            
            bb_position = (current_price - featured_data['BB_lower'].iloc[-1]) / (featured_data['BB_upper'].iloc[-1] - featured_data['BB_lower'].iloc[-1])
            bb_risk = "HIGH" if bb_position > 0.9 or bb_position < 0.1 else "MEDIUM" if bb_position > 0.7 or bb_position < 0.3 else "LOW"
            st.metric("Bollinger Risk", bb_risk, f"{bb_position:.2%}")
        
        with col3:
            st.markdown("#### Volume Risk")
            vol_ratio = featured_data['Volume'].iloc[-1] / featured_data['Volume_SMA'].iloc[-1]
            volume_risk = "HIGH" if vol_ratio > 2 else "MEDIUM" if vol_ratio > 1.5 else "LOW"
            st.metric("Volume Anomaly", volume_risk, f"{vol_ratio:.2f}x")
            
            mfi_val = featured_data['MFI'].iloc[-1]
            mfi_risk = "HIGH" if mfi_val > 80 or mfi_val < 20 else "MEDIUM" if mfi_val > 70 or mfi_val < 30 else "LOW"
            st.metric("MFI Risk", mfi_risk, f"{mfi_val:.2f}")
        
        # RSI chart
        st.markdown("### 📊 RSI Trend (Momentum Risk)")
        fig_rsi = plot_rsi(featured_data)
        st.pyplot(fig_rsi)
        
        # Technical signals summary
        st.markdown("### 📊 Technical Signal Summary")
        signal_cols = st.columns(4)
        with signal_cols[0]:
            st.metric("MACD", 
                     "Bullish ✅" if signals['MACD_Signal'].iloc[-1] > 0 else "Bearish ⚠️")
        with signal_cols[1]:
            rsi_signal = "Oversold ⬇️" if featured_data['RSI'].iloc[-1] < 30 else "Overbought ⬆️" if featured_data['RSI'].iloc[-1] > 70 else "Neutral ➡️"
            st.metric("RSI", rsi_signal)
        with signal_cols[2]:
            st.metric("MFI", featured_data['MFI'].iloc[-1].round(2))
        with signal_cols[3]:
            st.metric("Volume", 
                     "Above Avg ⬆️" if signals['Volume_Signal'].iloc[-1] > 0 else "Below Avg ⬇️")
        
        footer_set()
    
    # Tab 3: Risk Assessment
    with tabs[2]:
        st.markdown("## 🎯 Downside Risk Assessment")
        
        with st.spinner("🔄 Training ensemble models and calculating risk..."):
            try:
                # Train ensemble and get risk assessment
                ensemble_results = train_ensemble(featured_data)
                
                if ensemble_results is None:
                    st.warning("⚠️ Insufficient data for risk assessment. Please select a longer time period.")
                else:
                    current_data = get_next_day_data(featured_data)
                    risk_assessment = calculate_risk_score(ensemble_results, current_data, featured_data, current_price)
                    
                    if risk_assessment:
                        st.session_state['risk_scores'][company_symbol] = risk_assessment
                        
                        # Main Risk Score Display
                        st.markdown("### 🎯 Risk Score Summary")
                        
                        risk_score = risk_assessment['risk_score']
                        risk_class = risk_assessment['risk_class']
                        confidence = risk_assessment['confidence']
                        
                        # Determine risk color
                        if risk_class == "LOW RISK":
                            risk_color = "risk-low"
                            emoji = "✅"
                        elif risk_class == "MEDIUM RISK":
                            risk_color = "risk-medium"
                            emoji = "⚠️"
                        else:
                            risk_color = "risk-high"
                            emoji = "🔴"
                        
                        # Risk gauge display
                        st.markdown(f"""
                        <div class="risk-gauge {risk_color}">
                            <h1>{emoji} {risk_class}</h1>
                            <h2>Risk Score: {risk_score:.1f}/100</h2>
                            <h3>Confidence: {confidence:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk metrics in columns
                        st.markdown("### 📊 Risk Breakdown")
                        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                        
                        with risk_col1:
                            st.metric(
                                "Downside Probability",
                                f"{risk_assessment['downside_prob']:.1f}%",
                                "Potential Loss Risk"
                            )
                        
                        with risk_col2:
                            st.metric(
                                "Expected Downside",
                                f"{risk_assessment['expected_downside']:.2f}%",
                                "Price Impact"
                            )
                        
                        with risk_col3:
                            st.metric(
                                "Model Consensus",
                                f"{risk_assessment['model_agreement']:.1f}%",
                                "Agreement Level"
                            )
                        
                        with risk_col4:
                            predicted_price = risk_assessment.get('predicted_price', current_price)
                            price_change_pred = ((predicted_price - current_price) / current_price) * 100
                            st.metric(
                                "Price Projection",
                                f"${predicted_price:.2f}",
                                f"{price_change_pred:.2f}%"
                            )
                        
                        # Model Voting Results
                        st.markdown("### 🤖 Multi-Model Ensemble Voting")
                        st.markdown("**8 Independent ML Models Consensus:**")
                        
                        model_votes = risk_assessment['model_votes']
                        vote_col1, vote_col2 = st.columns(2)
                        
                        with vote_col1:
                            for i, (model_name, vote) in enumerate(list(model_votes.items())[:4]):
                                vote_emoji = "🔴" if vote == "HIGH" else "⚠️" if vote == "MEDIUM" else "✅"
                                st.markdown(f"**{model_name}:** {vote_emoji} {vote}")
                        
                        with vote_col2:
                            for model_name, vote in list(model_votes.items())[4:]:
                                vote_emoji = "🔴" if vote == "HIGH" else "⚠️" if vote == "MEDIUM" else "✅"
                                st.markdown(f"**{model_name}:** {vote_emoji} {vote}")
                        
                        # Risk Factor Contributions
                        st.markdown("### 🔍 Why This Risk Score? (Feature Importance)")
                        st.markdown("**Top Risk Contributors:**")
                        
                        contributions = risk_assessment['risk_contributions']
                        contrib_col1, contrib_col2 = st.columns(2)
                        
                        with contrib_col1:
                            for factor, contribution in list(contributions.items())[:3]:
                                st.markdown(f"**{factor}:** {contribution:.1f}% contribution")
                                st.progress(contribution / 100)
                        
                        with contrib_col2:
                            for factor, contribution in list(contributions.items())[3:6]:
                                st.markdown(f"**{factor}:** {contribution:.1f}% contribution")
                                st.progress(contribution / 100)
                        
                        # Risk Explanation
                        st.markdown("### 💡 Risk Explanation")
                        explanation = risk_assessment['explanation']
                        
                        for reason in explanation:
                            if "high" in reason.lower() or "bearish" in reason.lower():
                                st.error(f"⚠️ {reason}")
                            elif "medium" in reason.lower():
                                st.warning(f"⚡ {reason}")
                            else:
                                st.info(f"ℹ️ {reason}")
                        
                        # Risk Visualization
                        st.markdown("### 📈 Risk vs Time Projection")
                        try:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Historical prices
                            ax.plot(data['Date'], data['Close'], 
                                label='Historical Price', color='#1f77b4', linewidth=2)
                            
                            # Projected price with risk bands
                            if 'Short' in prediction_term:
                                future_date = data['Date'].iloc[-1] + pd.Timedelta(minutes=days_to_predict)
                            elif 'Mid' in prediction_term:
                                future_date = data['Date'].iloc[-1] + pd.Timedelta(hours=days_to_predict)
                            else:
                                future_date = data['Date'].iloc[-1] + pd.Timedelta(days=days_to_predict)
                            
                            predicted_price = risk_assessment.get('predicted_price', current_price)
                            
                            ax.plot([data['Date'].iloc[-1], future_date], 
                                [data['Close'].iloc[-1], predicted_price],
                                label='Projected Path', color='#ff7f0e', 
                                linewidth=2, linestyle='--')
                            
                            # Risk bands
                            downside = predicted_price * (1 - risk_assessment['expected_downside'] / 100)
                            upside = predicted_price * 1.02  # Small upside potential
                            
                            ax.fill_between([data['Date'].iloc[-1], future_date],
                                        [data['Close'].iloc[-1], downside],
                                        [data['Close'].iloc[-1], upside],
                                        color='red' if risk_class == "HIGH RISK" else 'orange' if risk_class == "MEDIUM RISK" else 'green',
                                        alpha=0.2, label='Risk Range')
                            
                            ax.set_title(f'{company_symbol} - Risk-Adjusted Price Projection', fontsize=14)
                            ax.set_xlabel('Date', fontsize=12)
                            ax.set_ylabel('Price ($)', fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating risk visualization: {str(e)}")
                            logging.error(f"Risk visualization error: {e}")
                        
                        # Investment Recommendation
                        st.markdown("### 💼 Investment Guidance")
                        if risk_class == "LOW RISK":
                            st.success("""
                            ✅ **Low Risk Assessment**
                            - Favorable technical indicators
                            - Low probability of significant downside
                            - Consider this as a potential opportunity
                            - Still conduct thorough due diligence
                            """)
                        elif risk_class == "MEDIUM RISK":
                            st.warning("""
                            ⚠️ **Medium Risk Assessment**
                            - Mixed technical signals
                            - Moderate downside probability
                            - Exercise caution
                            - Consider risk management strategies
                            """)
                        else:
                            st.error("""
                            🔴 **High Risk Assessment**
                            - Unfavorable technical indicators detected
                            - Elevated probability of downside movement
                            - High caution advised
                            - Consider avoiding or hedging position
                            """)
                        
                        st.info("⚠️ **Disclaimer:** This risk assessment is based on technical analysis and machine learning models. It should not be the sole basis for investment decisions. Always conduct comprehensive research and consult financial advisors.")
                    
                    else:
                        st.error("❌ Unable to generate risk assessment")
            
            except Exception as e:
                st.error(f"❌ Error in risk assessment: {str(e)}")
                logging.error(f"Risk assessment error: {e}")
        
        footer_set()
    
    # Tab 4: Data & Metrics
    with tabs[3]:
        st.markdown("## 📑 Data & Analysis")
        
        # Data overview
        st.markdown("### 📊 Dataset Overview")
        
        st.markdown(f"""
        **Dataset Characteristics:**
        - Total Samples: {len(featured_data)}
        - Features: {len(featured_data.columns)}
        - Time Period: {data['Date'].iloc[0].strftime('%Y-%m-%d')} to {data['Date'].iloc[-1].strftime('%Y-%m-%d')}
        - Interval: {interval}
        """)
        
        # Summary statistics
        st.markdown("#### Summary Statistics")
        summary_stats = featured_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
        st.dataframe(summary_stats.style.format("{:.2f}"))
        
        # Risk factors data
        st.markdown("#### Risk Factors (Structured Features)")
        risk_features = ['RSI', 'MACD', 'ATR', 'BB_middle', 'OBV', 'Pivot', 'MFI', 'ma_10']
        available_features = [f for f in risk_features if f in featured_data.columns]
        
        if available_features:
            st.dataframe(
                featured_data[['Date'] + available_features].tail(20).style.format("{:.2f}")
            )
        
        # Performance metrics
        st.markdown("### 📈 Statistical Risk Metrics")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            returns = featured_data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            st.metric("Volatility (Annualized)", f"{volatility:.2%}")
        
        with perf_col2:
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        with perf_col3:
            max_drawdown = (featured_data['Close'] / featured_data['Close'].cummax() - 1).min()
            st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
        
        # Export data option
        st.markdown("### 💾 Export Data")
        if st.button("Download Risk Analysis Data"):
            csv = featured_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{company_symbol}_risk_analysis.csv",
                mime="text/csv"
            )
        
        footer_set()
    
    # Cleanup
    progress_bar.empty()
    status_text.empty()

else:
    st.info("👈 Please enter a stock symbol in the sidebar to begin risk analysis")
    
    # Show example
    st.markdown("""
    ### 🎯 About This Risk Predictor
    
    This system uses **8 classic machine learning models** trained on **structured market data** to predict downside risk:
    
    **Features (12+ Risk Factors):**
    - Momentum: RSI, MACD, ROC
    - Volatility: ATR, Bollinger Bands
    - Volume: OBV, Volume Delta
    - Trend: Moving Averages, Trend Slope
    - Support/Resistance Levels
    - Pattern Signals
    
    **Models Used:**
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. Random Forest
    5. Gradient Boosting
    6. Support Vector Regression
    7. XGBoost
    8. CatBoost
    
    **Output:**
    - Risk Score (0-100)
    - Risk Class (Low/Medium/High)
    - Model Consensus
    - Feature Importance
    - Interpretable Explanations
    
    **Perfect for PS-17:** Complete ML workflow with structured data, model diversity, and interpretability!
    """)
    
    footer_set()