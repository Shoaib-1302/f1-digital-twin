import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os

# Setup Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Add both src and project root to path
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

# Try to import utilities
try:
    from utils.helpers import load_config, initialize_session_state
    from utils.visualizations import create_header
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    # Fallback functions
    def load_config():
        return {
            'app': {'name': 'F1 Digital Twin', 'version': '1.0.0'},
            'features': {
                'enable_predictions': False,
                'enable_rag_insights': False,
                'enable_news_collection': False
            }
        }
    
    def initialize_session_state():
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
    
    def create_header(title, subtitle="", icon="🏎️"):
        return f"""
        <div style='background: linear-gradient(90deg, #E10600 0%, #FF1E00 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h1 style='margin: 0;'>{icon} {title}</h1>
            {f"<p style='font-size: 1.2rem; margin-top: 0.5rem;'>{subtitle}</p>" if subtitle else ""}
        </div>
        """

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
    }
    .stButton>button {
        background-color: #E10600;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #FF1E00;
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
    
    # Check for data
    data_available = False
    data_paths = [
        'data/processed/processed_data.csv',
        'data/raw/results.csv'
    ]
    
    for path in data_paths:
        if Path(path).exists():
            data_available = True
            break
    
    if not data_available:
        st.warning("⚠️ No data found!")
        st.info("""
        ### Getting Started
        
        To use this application, you need to collect F1 data:
        
        **Option 1: Run data collection locally**
        ```bash
        python -m src.data.data_collector --start-year 2020 --end-year 2024
        python -m src.data.data_preprocessor
        ```
        
        **Option 2: Upload your data**
        Use the file uploader in each page to upload CSV files.
        
        **Option 3: Use sample data**
        Download sample data from the repository.
        """)
    
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
        - **OpenF1 API**: Real-time and recent race data
        - **Kaggle F1 Dataset**: Comprehensive historical records
        
        #### Contextual Textual Data:
        
        - **NewsAPI.org**: Recent F1 news articles
        - **Formula1.com**: Official race reports and analysis
        - **Expert Blogs**: Technical analysis and commentary
        """)
    
    with tab3:
        st.markdown("""
        ### Technology Stack
        
        #### Core Models:
        
        1. **Temporal Fusion Transformer (TFT)**
           - Multi-horizon time-series forecasting
           - Built-in attention mechanisms for interpretability
        
        2. **Retrieval-Augmented Generation (RAG)**
           - Dense passage retrieval with embeddings
           - GPT-based natural language generation
        
        #### Framework & Libraries:
        
        - **Streamlit**: Interactive web application
        - **PyTorch**: Deep learning framework
        - **Transformers (Hugging Face)**: NLP models
        - **Plotly**: Interactive visualizations
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
           - Upload data or use collected data
           - View performance trends
        
        2. **Check Team Comparison** 🏆
           - Visit Team Comparison page
           - Analyze constructor performance
           - Track development trends
        """)
    
    with col2:
        st.markdown("""
        ### System Status
        
        """)
        
        # Show status
        st.success("✅ Application running") if True else st.error("❌ Error")
        st.info(f"📊 Data available: {'Yes' if data_available else 'No'}")
        st.info(f"🔧 Utils available: {'Yes' if UTILS_AVAILABLE else 'No'}")
    
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
