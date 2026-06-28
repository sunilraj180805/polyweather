"""
PolyWeather — Streamlit Dashboard

Interactive UI to monitor the autonomous trading system.
Provides 4 tabs:
1. Active Positions
2. Trade History & PnL
3. Weather Metrics vs Predictions
4. Risk Profiles
"""

import sqlite3
from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from execution_engine import PaperTrader
from hermes_orchestrator import HermesOrchestrator
from data_engine import WeatherDataEngine
from prediction_model import WeatherPredictor
from market_data import PolymarketFetcher

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PolyWeather Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize engines
@st.cache_resource
def get_engines():
    trader = PaperTrader()
    data_engine = WeatherDataEngine()
    predictor = WeatherPredictor(data_engine)
    fetcher = PolymarketFetcher()
    return trader, data_engine, predictor, fetcher

trader, data_engine, predictor, fetcher = get_engines()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
    }
    .status-active { color: #4CAF50; font-weight: bold; }
    .status-resolved { color: #A0A0A0; font-weight: bold; }
    .side-yes { color: #4CAF50; font-weight: bold; }
    .side-no { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def load_trades() -> pd.DataFrame:
    """Load full trade history from SQLite into a DataFrame."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load trades: {e}")
        return pd.DataFrame()

def get_portfolio_metrics() -> dict:
    return trader.get_portfolio()

def get_pnl_metrics() -> dict:
    return trader.get_pnl_summary()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🌤️ PolyWeather")
    st.markdown("Autonomous Weather Trading")
    st.markdown("---")
    
    pnl = get_pnl_metrics()
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>${pnl.get('current_balance', 0):,.2f}</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Current Balance</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    color = "#4CAF50" if pnl.get("total_return_pct", "0%").startswith("-") == False else "#F44336"
    st.markdown(f"<div class='metric-value' style='color:{color}'>{pnl.get('total_return_pct', '0%')}</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Total Return</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{pnl.get('win_rate_pct', '0%')}</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>Win Rate</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Active Positions", 
    "📈 Trade History & PnL", 
    "🌤️ Weather Predictions", 
    "⚠️ Risk Profiles"
])

df_trades = load_trades()

# --- Tab 1: Active Positions ---
with tab1:
    st.header("Active Positions")
    portfolio = get_portfolio_metrics()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invested", f"${portfolio.get('total_invested', 0):,.2f}")
    col2.metric("Open Positions", portfolio.get('num_positions', 0))
    col3.metric("Total Exposure Limit", f"{config.MAX_TOTAL_EXPOSURE_PCT * 100}%")

    if not df_trades.empty:
        open_trades = df_trades[df_trades['status'] == 'FILLED'].copy()
        if not open_trades.empty:
            open_trades['Value'] = open_trades['amount']
            open_trades = open_trades[['timestamp', 'market_id', 'city', 'event_type', 'side', 'amount', 'price', 'shares']]
            
            # Formatting
            st.dataframe(
                open_trades.style.format({
                    'amount': '${:,.2f}',
                    'price': '{:.3f}',
                    'shares': '{:,.2f}'
                }, na_rep="").map(lambda x: 'color: #4CAF50' if x == 'YES' else 'color: #F44336', subset=['side']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No active positions.")
    else:
        st.info("No trade data available.")


# --- Tab 2: Trade History & PnL ---
with tab2:
    st.header("Trade History & Cumulative PnL")
    
    if not df_trades.empty:
        # PnL Chart
        resolved_trades = df_trades[df_trades['status'] == 'RESOLVED'].copy()
        if not resolved_trades.empty:
            resolved_trades['timestamp'] = pd.to_datetime(resolved_trades['timestamp'])
            resolved_trades = resolved_trades.sort_values('timestamp')
            
            # Cumulative PnL starting from initial balance
            resolved_trades['cumulative_pnl'] = resolved_trades['pnl'].cumsum()
            resolved_trades['equity'] = config.INITIAL_BALANCE + resolved_trades['cumulative_pnl']
            
            # Create a starting point
            start_row = pd.DataFrame({
                'timestamp': [resolved_trades['timestamp'].iloc[0] - pd.Timedelta(hours=1)],
                'equity': [config.INITIAL_BALANCE]
            })
            chart_data = pd.concat([start_row, resolved_trades[['timestamp', 'equity']]])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_data['timestamp'], 
                y=chart_data['equity'],
                mode='lines+markers',
                name='Equity',
                line=dict(color='#4CAF50', width=3),
                fill='tozeroy',
                fillcolor='rgba(76, 175, 80, 0.2)'
            ))
            fig.update_layout(
                title="Cumulative Equity Growth",
                xaxis_title="Time",
                yaxis_title="Account Balance ($)",
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough resolved trades to generate a PnL chart.")

        # Full History Table
        st.subheader("All Trades")
        display_df = df_trades[['id', 'timestamp', 'city', 'event_type', 'side', 'amount', 'price', 'status', 'pnl']].copy()
        
        # Color coding
        def color_pnl(val):
            if pd.isna(val): return ''
            color = '#4CAF50' if val > 0 else '#F44336' if val < 0 else '#A0A0A0'
            return f'color: {color}'
            
        st.dataframe(
            display_df.style.format({
                'amount': '${:,.2f}',
                'price': '{:.3f}',
                'pnl': '${:,.2f}'
            }, na_rep="").map(color_pnl, subset=['pnl'])
              .map(lambda x: 'color: #4CAF50' if x == 'YES' else 'color: #F44336' if x == 'NO' else '', subset=['side']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trade data available.")


# --- Tab 3: Weather vs Predictions ---
with tab3:
    st.header("Live Weather & Prediction Matrices")
    
    try:
        tomorrow = (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        st.subheader(f"Forecast Target: {tomorrow}")
        
        weather_data = data_engine.fetch_all_cities()
        predictions = predictor.predict_all_cities(target_date=tomorrow)
        
        # Build comparison grid
        cols = st.columns(len(config.CITIES))
        for i, (city_key, city_cfg) in enumerate(config.CITIES.items()):
            with cols[i]:
                st.markdown(f"### {city_cfg['display_name']}")
                
                # Weather Summary
                city_w = weather_data.get(city_key, {})
                cur = city_w.get("current", {})
                st.markdown(f"**Current**: {cur.get('temp_f', '?')}°F, {cur.get('condition', '?')}")
                
                # Model Predictions
                st.markdown("**Model Probabilities:**")
                city_preds = predictions.get(city_key, {}).get("predictions", {})
                
                for ev_key, p_data in city_preds.items():
                    if ev_key == "temp_range_buckets": continue
                    
                    prob = p_data.get("probability", 0)
                    prob_pct = f"{prob*100:.1f}%"
                    
                    # Highlight high confidence
                    if prob > 0.8:
                        color = "#4CAF50"
                    elif prob < 0.2:
                        color = "#F44336"
                    else:
                        color = "#A0A0A0"
                        
                    st.markdown(f"- {ev_key}: <span style='color:{color}; font-weight:bold;'>{prob_pct}</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Failed to load prediction matrices: {e}")


# --- Tab 4: Risk Profiles ---
with tab4:
    st.header("Risk Management Profiles")
    
    from risk_manager import RiskManager
    rm = RiskManager(bankroll=pnl.get('current_balance', config.INITIAL_BALANCE))
    
    if not df_trades.empty:
        open_pos = trader.get_open_positions()
        risk_report = rm.check_portfolio_risk(open_pos)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Exposure Distribution")
            city_exp = risk_report.city_exposure
            if city_exp:
                fig = go.Figure(data=[go.Pie(labels=list(city_exp.keys()), values=list(city_exp.values()), hole=.3)])
                fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No exposure currently.")
                
        with col2:
            st.subheader("Safety Parameters")
            st.metric("Total Exposure", f"{risk_report.exposure_pct*100:.1f}%", f"Limit: {config.MAX_TOTAL_EXPOSURE_PCT*100}%", delta_color="inverse")
            st.metric("Max Single Position", f"{risk_report.max_single_pct*100:.1f}%", f"Limit: {config.MAX_SINGLE_POSITION_PCT*100}%", delta_color="inverse")
            st.metric("Kelly Multiplier Cap", f"{config.KELLY_FRACTION_CAP}x (Half-Kelly)")
            
            if not risk_report.within_limits:
                st.error("⚠️ PORTFOLIO EXCEEDS RISK LIMITS")
                for w in risk_report.warnings:
                    st.warning(w)
            else:
                st.success("✅ Portfolio operating within configured risk limits")
                
        # Hedging Suggestions
        st.subheader("Hedging Recommendations")
        hedges = rm.suggest_hedges(open_pos)
        if hedges:
            for h in hedges:
                st.info(f"💡 {h.reason}")
        else:
            st.success("No critical hedges required at current exposure levels.")
            
    else:
        st.info("No active positions to analyze.")
