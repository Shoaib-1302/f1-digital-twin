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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.visualizations import (
    create_performance_timeline,
    create_prediction_chart,
    create_confidence_interval_chart
)
from utils.helpers import load_driver_data, format_prediction_text
from models.predictor import F1PerformancePredictor
from models.rag_pipeline import F1RAGPipeline


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
        # Load from session state or files
        if 'driver_data' not in st.session_state:
            st.session_state.driver_data = pd.read_csv('data/processed/driver_performance.csv')
        
        return st.session_state.driver_data
    except FileNotFoundError:
        st.warning("Data not found. Please run data collection first.")
        return None


def main():
    st.title("🏎️ Driver Performance Analysis")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Unable to load driver data. Please check data collection.")
        return
    
    # Sidebar - Driver Selection
    st.sidebar.header("Select Driver")
    
    # Get unique drivers
    drivers = sorted(data['driver_name'].unique())
    selected_driver = st.sidebar.selectbox(
        "Driver",
        drivers,
        index=0
    )
    
    # Get driver ID
    driver_id = data[data['driver_name'] == selected_driver]['driver_id'].iloc[0]
    
    # Filter data for selected driver
    driver_data = data[data['driver_id'] == driver_id].copy()
    driver_data = driver_data.sort_values('race_index')
    
    # Analysis options
    st.sidebar.header("Analysis Options")
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    show_insights = st.sidebar.checkbox("Show AI Insights", value=True)
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
    current_season = driver_data['season'].max()
    season_data = driver_data[driver_data['season'] == current_season]
    
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
        
        # Performance over time chart
        fig = go.Figure()
        
        # Points per race
        fig.add_trace(go.Scatter(
            x=driver_data['race_name'],
            y=driver_data['points'],
            mode='lines+markers',
            name='Points',
            line=dict(color='#E10600', width=2),
            marker=dict(size=8)
        ))
        
        # Add rolling average
        if len(driver_data) >= 5:
            driver_data['rolling_avg'] = driver_data['points'].rolling(window=5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=driver_data['race_name'],
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
            x=driver_data['race_name'],
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
            yaxis=dict(autorange='reversed'),  # Lower position is better
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_pos, use_container_width=True)
    
    with tab2:
        if show_predictions:
            st.subheader("🔮 Performance Predictions")
            
            # Load model
            try:
                with st.spinner("Loading prediction model..."):
                    predictor = F1PerformancePredictor(
                        model_path=Path('data/models/tft_model.pt')
                    )
                
                # Make predictions
                with st.spinner("Generating predictions..."):
                    predictions = predictor.predict_next_races(
                        driver_id=driver_id,
                        historical_data=driver_data,
                        n_races=n_races_predict
                    )
                
                # Display predictions
                st.markdown("### Next Race Predictions")
                
                for idx, row in predictions.iterrows():
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4 style='margin-top: 0;'>Race +{row['race_number']}</h4>
                            <p style='font-size: 1.5rem; margin: 0.5rem 0;'>
                                <strong>{row['predicted_points']:.1f} Points</strong>
                            </p>
                            <p style='color: #666; margin: 0;'>
                                80% Confidence: {row['confidence_80_lower']:.1f} - {row['confidence_80_upper']:.1f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Mini gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=row['predicted_points'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 26]},
                                'bar': {'color': "#E10600"},
                                'steps': [
                                    {'range': [0, 10], 'color': "lightgray"},
                                    {'range': [10, 18], 'color': "gray"}
                                ]
                            }
                        ))
                        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Prediction chart with confidence intervals
                st.markdown("### Prediction Trend")
                
                fig_pred = go.Figure()
                
                races = list(range(1, n_races_predict + 1))
                
                # Confidence intervals
                fig_pred.add_trace(go.Scatter(
                    x=races + races[::-1],
                    y=list(predictions['confidence_95_upper']) + list(predictions['confidence_95_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(225, 6, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence',
                    showlegend=True
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=races + races[::-1],
                    y=list(predictions['confidence_80_upper']) + list(predictions['confidence_80_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(225, 6, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='80% Confidence',
                    showlegend=True
                ))
                
                # Median prediction
                fig_pred.add_trace(go.Scatter(
                    x=races,
                    y=predictions['predicted_points'],
                    mode='lines+markers',
                    name='Predicted Points',
                    line=dict(color='#E10600', width=3),
                    marker=dict(size=10)
                ))
                
                fig_pred.update_layout(
                    title="Predicted Performance with Confidence Intervals",
                    xaxis_title="Race Number",
                    yaxis_title="Points",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # AI Insights
                if show_insights:
                    st.markdown("### 🤖 AI-Generated Insights")
                    
                    with st.spinner("Analyzing recent news and context..."):
                        try:
                            rag = F1RAGPipeline(
                                corpus_path=Path('data/news_corpus')
                            )
                            
                            # Get explanation for first prediction
                            pred_dict = predictions.iloc[0].to_dict()
                            insights = rag.explain_prediction(
                                driver_name=selected_driver,
                                prediction=pred_dict
                            )
                            
                            st.info(insights['explanation'])
                            
                            # Show sources
                            with st.expander("📰 View Sources"):
                                for source in insights['sources']:
                                    st.markdown(f"""
                                    **{source['title']}**  
                                    *{source['source']} - {source['date']}*  
                                    Relevance: {source['relevance_score']:.2%}
                                    """)
                        
                        except Exception as e:
                            st.warning(f"Unable to generate AI insights: {str(e)}")
            
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please ensure the model is trained and saved in data/models/")
        
        else:
            st.info("Enable 'Show Predictions' in the sidebar to see forecasts.")
    
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
        recent_races = driver_data.tail(10)[
            ['race_name', 'date', 'grid', 'position', 'points', 'status']
        ].sort_values('date', ascending=False)
        
        st.dataframe(
            recent_races,
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()
