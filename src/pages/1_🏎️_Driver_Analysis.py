"""
Driver Analysis Page
Individual driver performance analysis and predictions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
from _imports import *

# Fix imports for Streamlit Cloud
if os.path.exists('/mount/src/f1-digital-twin'):
    # Running on Streamlit Cloud
    sys.path.insert(0, '/mount/src/f1-digital-twin/src')
else:
    # Running locally
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.visualizations import (
        create_performance_timeline,
        create_prediction_chart,
        create_confidence_interval_chart
    )
    from utils.helpers import load_driver_data, format_prediction_text
    from models.predictor import F1PerformancePredictor
    from models.rag_pipeline import F1RAGPipeline
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Some features may not be available. Please ensure all modules are installed.")
    # Create dummy functions to prevent crashes
    def create_performance_timeline(*args, **kwargs):
        return go.Figure()
    def create_prediction_chart(*args, **kwargs):
        return go.Figure()
    F1PerformancePredictor = None
    F1RAGPipeline = None


st.set_page_config(
    page_title="Driver Analysis",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .driver-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E10600;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load driver and performance data"""
    try:
        # Try multiple possible data locations
        possible_paths = [
            'data/processed/driver_performance.csv',
            'data/processed/processed_data.csv',
            'data/raw/results.csv',
            '/mount/src/f1-digital-twin/data/processed/processed_data.csv',
            '/mount/src/f1-digital-twin/data/raw/results.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                st.session_state.driver_data = df
                return df
        
        st.warning("Data files not found. Please run data collection first.")
        return None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main():
    st.title("🏎️ Driver Performance Analysis")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Unable to load driver data.")
        st.info("""
        ### Getting Started
        
        To use this page, you need to:
        1. Collect F1 data: `python -m src.data.data_collector`
        2. Preprocess data: `python -m src.data.data_preprocessor`
        3. Refresh this page
        
        Or upload your own data using the file uploader below.
        """)
        
        # File uploader as fallback
        uploaded_file = st.file_uploader("Upload driver data (CSV)", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
        else:
            return
    
    # Ensure required columns exist
    required_cols = ['driver_id', 'driver_name', 'points', 'position']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.info(f"Available columns: {', '.join(data.columns.tolist())}")
        return
    
    # Sidebar - Driver Selection
    st.sidebar.header("Select Driver")
    
    # Get unique drivers
    drivers = sorted(data['driver_name'].unique())
    
    if len(drivers) == 0:
        st.error("No drivers found in data")
        return
    
    selected_driver = st.sidebar.selectbox(
        "Driver",
        drivers,
        index=0
    )
    
    # Get driver ID
    driver_id = data[data['driver_name'] == selected_driver]['driver_id'].iloc[0]
    
    # Filter data for selected driver
    driver_data = data[data['driver_id'] == driver_id].copy()
    
    # Sort by date or race index
    if 'date' in driver_data.columns:
        driver_data['date'] = pd.to_datetime(driver_data['date'])
        driver_data = driver_data.sort_values('date')
    elif 'race_index' in driver_data.columns:
        driver_data = driver_data.sort_values('race_index')
    
    # Analysis options
    st.sidebar.header("Analysis Options")
    show_predictions = st.sidebar.checkbox("Show Predictions", value=False)  # Disabled by default
    show_insights = st.sidebar.checkbox("Show AI Insights", value=False)
    n_races_predict = st.sidebar.slider("Races to Predict", 1, 5, 3)
    
    # Driver Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="driver-card">
            <h2>{selected_driver}</h2>
            <p style='font-size: 1.2rem;'>Performance Analysis & Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Current season stats
    if 'season' in driver_data.columns:
        current_season = driver_data['season'].max()
        season_data = driver_data[driver_data['season'] == current_season]
    else:
        season_data = driver_data
        current_season = "All Time"
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_points = season_data['points'].sum()
        st.markdown(f"""
        <div class="metric-box">
            <h3 style='color: #E10600; margin: 0;'>{total_points:.0f}</h3>
            <p style='color: #666; margin: 0;'>Total Points ({current_season})</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_position = season_data['position'].mean()
        st.markdown(f"""
        <div class="metric-box">
            <h3 style='color: #E10600; margin: 0;'>P{avg_position:.1f}</h3>
            <p style='color: #666; margin: 0;'>Avg Position</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        podiums = len(season_data[season_data['position'] <= 3])
        st.markdown(f"""
        <div class="metric-box">
            <h3 style='color: #E10600; margin: 0;'>{podiums}</h3>
            <p style='color: #666; margin: 0;'>Podiums</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        wins = len(season_data[season_data['position'] == 1])
        st.markdown(f"""
        <div class="metric-box">
            <h3 style='color: #E10600; margin: 0;'>{wins}</h3>
            <p style='color: #666; margin: 0;'>Wins</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "📈 Performance Timeline",
        "🔮 Predictions",
        "📊 Statistics"
    ])
    
    with tab1:
        st.subheader("Historical Performance")
        
        # Points per race chart
        fig = go.Figure()
        
        x_values = driver_data['race_name'] if 'race_name' in driver_data.columns else range(len(driver_data))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=driver_data['points'],
            mode='lines+markers',
            name='Points',
            line=dict(color='#E10600', width=2),
            marker=dict(size=8)
        ))
        
        # Add rolling average if enough data
        if len(driver_data) >= 5:
            driver_data['rolling_avg'] = driver_data['points'].rolling(window=5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=driver_data['rolling_avg'],
                mode='lines',
                name='5-Race Average',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Points per Race",
            xaxis_title="Race",
            yaxis_title="Points",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Position chart
        fig_pos = go.Figure()
        
        fig_pos.add_trace(go.Scatter(
            x=x_values,
            y=driver_data['position'],
            mode='lines+markers',
            name='Position',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=8)
        ))
        
        fig_pos.update_layout(
            title="Race Position",
            xaxis_title="Race",
            yaxis_title="Position",
            yaxis=dict(autorange='reversed'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with tab2:
        if show_predictions and F1PerformancePredictor is not None:
            st.subheader("🔮 Performance Predictions")
            st.info("Prediction features require model training. Please train the model first.")
        else:
            st.info("""
            ### Predictions Not Available
            
            To enable predictions:
            1. Train the TFT model: `python -m src.models.train`
            2. Enable "Show Predictions" in the sidebar
            
            Or you can view historical statistics in the Statistics tab.
            """)
    
    with tab3:
        st.subheader("📊 Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Points Distribution")
            fig_hist = px.histogram(
                driver_data,
                x='points',
                nbins=20,
                title="Points per Race Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### Position Distribution")
            fig_pos_hist = px.histogram(
                driver_data,
                x='position',
                nbins=20,
                title="Finishing Position Distribution"
            )
            st.plotly_chart(fig_pos_hist, use_container_width=True)
        
        # Recent races table
        st.markdown("#### Recent Races")
        
        display_cols = ['race_name', 'position', 'points']
        if 'date' in driver_data.columns:
            display_cols.insert(1, 'date')
        if 'grid' in driver_data.columns:
            display_cols.insert(2, 'grid')
        if 'status' in driver_data.columns:
            display_cols.append('status')
        
        available_cols = [col for col in display_cols if col in driver_data.columns]
        recent_races = driver_data[available_cols].tail(10).sort_values(
            'date' if 'date' in available_cols else available_cols[0],
            ascending=False
        )
        
        st.dataframe(
            recent_races,
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()
