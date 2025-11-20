"""
F1 Digital Twin - Main Streamlit Application
Entry point for the Formula 1 behavioral digital twin system
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.helpers import load_config, initialize_session_state
from utils.visualizations import create_header

# Page configuration
st.set_page_config(
    page_title="F1 Digital Twin",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #E10600 0%, #FF1E00 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #E10600;
        color: black !important;   /* <-- Added */
    }

    .stButton>button {
        background-color: #E10600;
        color: black;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #FF1E00;
    }

    /* NEW GLOBAL TEXT COLOR FIX */
    body, p, div, span, h1, h2, h3, h4, h5, h6, label {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏎️ Formula 1 Digital Twin System</h1>
        <p style='font-size: 1.2rem; margin-top: 0.5rem;'>
            AI-Powered Performance Prediction & Race Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Driver Analysis</h3>
            <p>Predict individual driver performance using historical data and AI models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏁 Race Predictions</h3>
            <p>Forecast race outcomes with Temporal Fusion Transformer models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📰 AI Insights</h3>
            <p>Get contextual explanations powered by RAG and news analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview section
    st.header("📊 System Overview")
    
    tab1, tab2, tab3 = st.tabs(["About", "Data Sources", "Technology"])
    
    with tab1:
        st.markdown("""
        ### What is F1 Digital Twin?
        
        This system creates **behavioral digital twins** of Formula 1 drivers and teams by combining:
        
        - **Time-series performance data**: Historical race results, lap times, telemetry
        - **Contextual knowledge**: News articles, expert commentary, injury reports
        - **AI models**: Temporal Fusion Transformer for predictions, RAG for explanations
        
        #### Key Features:
        
        1. **Driver Performance Prediction**: Forecast how drivers will perform in upcoming races
        2. **Team Dynamics Analysis**: Understand constructor strategies and comparative performance
        3. **Explainable AI**: Get natural language explanations for predictions
        4. **Real-time Updates**: Integration with live F1 data feeds
        
        #### Use Cases:
        
        - 🏎️ **Teams**: Strategic planning and tactical decision-making
        - 📺 **Broadcasters**: Enhanced fan engagement and commentary
        - 🎮 **Fantasy F1**: Data-driven team selection
        - 📊 **Analysts**: Deep performance insights
        """)
    
    with tab2:
        st.markdown("""
        ### Data Sources
        
        #### Time-Series Performance Data:
        
        - **Ergast Developer API**: Historical F1 data from 1950-present
          - Race results, qualifying times, lap data, pit stops
          - Driver standings, constructor standings
          - Circuit information and race schedules
        
        - **OpenF1 API**: Real-time and recent race data
          - Live timing and telemetry
          - Lap times and sector times
          - Tyre strategies and overtakes
        
        - **Kaggle F1 Dataset**: Comprehensive historical records
          - 1950-2024 championship data
          - Driver ratings and performance metrics
        
        #### Contextual Textual Data:
        
        - **NewsAPI.org**: Recent F1 news articles
        - **Formula1.com**: Official race reports and analysis
        - **Expert Blogs**: Technical analysis and commentary
        - **Injury Reports**: Driver fitness and medical updates
        """)
    
    with tab3:
        st.markdown("""
        ### Technology Stack
        
        #### Core Models:
        
        1. **Temporal Fusion Transformer (TFT)**
           - Multi-horizon time-series forecasting
           - Handles static, time-varying, and known future inputs
           - Built-in attention mechanisms for interpretability
        
        2. **Retrieval-Augmented Generation (RAG)**
           - Dense passage retrieval with embeddings
           - GPT-based natural language generation
           - Context-aware explanations
        
        #### Framework & Libraries:
        
        - **Streamlit**: Interactive web application
        - **PyTorch**: Deep learning framework
        - **Transformers (Hugging Face)**: NLP models
        - **Plotly**: Interactive visualizations
        - **Pandas/NumPy**: Data manipulation
        - **Scikit-learn**: ML utilities
        
        #### Infrastructure:
        
        - **Docker**: Containerized deployment
        - **PostgreSQL**: Data storage (optional)
        - **Redis**: Caching layer (optional)
        """)
    
    st.markdown("---")
    
    # Getting Started
    st.header("🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Quick Start Guide
        
        1. **Explore Driver Analysis** 👤
           - Navigate to the Driver Analysis page
           - Select a driver to see their performance trends
           - View AI-generated predictions
        
        2. **Check Race Predictions** 🏁
           - Go to Race Predictions page
           - See forecasts for upcoming races
           - Compare different scenarios
        
        3. **Compare Teams** 🏆
           - Visit Team Comparison page
           - Analyze constructor performance
           - Track development trends
        """)
    
    with col2:
        st.markdown("""
        ### Need Help?
        
        📖 **Documentation**: Check the README.md for detailed instructions
        
        🐛 **Issues**: Report bugs on GitHub
        
        💡 **Feature Requests**: Submit via GitHub Issues
        
        📧 **Contact**: For support and questions
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: 2024
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit, PyTorch, and the Transformers library</p>
        <p>Data provided by Ergast API, OpenF1 API, and NewsAPI.org</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
