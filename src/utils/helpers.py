"""
Utility Helper Functions
Common utilities used across the application
"""

import streamlit as st
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json


def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """Load application configuration from YAML file"""
    try:
        config_file = Path(config_path)
        
        # Check if config file exists
        if not config_file.exists():
            st.warning(f"Config file not found: {config_path}. Using defaults.")
            return get_default_config()
        
        # Load YAML
        with open(config_file, 'r') as f:
            # Read entire file content
            content = f.read()
            
            # Check if it's a multi-document YAML (contains ---)
            if '\n---\n' in content or content.startswith('---\n'):
                # Split into documents and load first one
                docs = content.split('\n---\n')
                config = yaml.safe_load(docs[0])
            else:
                # Single document, load normally
                config = yaml.safe_load(content)
        
        # Validate config
        if not isinstance(config, dict):
            st.warning("Invalid config format. Using defaults.")
            return get_default_config()
        
        return config
    
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML config: {e}")
        return get_default_config()
    
    except Exception as e:
        st.warning(f"Error loading config: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict:
    """Return default configuration"""
    return {
        'app': {
            'name': 'F1 Digital Twin',
            'version': '1.0.0',
            'debug': False
        },
        'data': {
            'start_year': 2018,
            'end_year': 2024,
            'min_races_for_prediction': 5,
            'paths': {
                'raw': 'data/raw',
                'processed': 'data/processed',
                'models': 'data/models',
                'news_corpus': 'data/news_corpus',
                'cache': 'data/cache'
            }
        },
        'visualization': {
            'theme': 'plotly_white',
            'chart_height': 500,
            'show_confidence_intervals': True
        },
        'model': {
            'hidden_size': 160,
            'attention_heads': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64
        },
        'cache': {
            'enabled': False,
            'ttl_seconds': 3600,
            'max_size_mb': 1000
        },
        'features': {
            'enable_predictions': True,
            'enable_rag_insights': True,
            'enable_news_collection': True,
            'enable_real_time_data': False
        }
    }


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.driver_data = None
        st.session_state.predictions = None
        st.session_state.current_season = 2024
        st.session_state.selected_driver = None
        st.session_state.model_loaded = False


def load_driver_data(driver_id: str, data_path: Path) -> pd.DataFrame:
    """Load historical data for a specific driver"""
    try:
        all_data = pd.read_csv(data_path)
        driver_data = all_data[all_data['driver_id'] == driver_id].copy()
        driver_data = driver_data.sort_values(['season', 'round'])
        return driver_data
    except Exception as e:
        st.error(f"Error loading driver data: {str(e)}")
        return pd.DataFrame()


def format_prediction_text(prediction: Dict, driver_name: str) -> str:
    """Format prediction data into readable text"""
    text = f"**{driver_name}** is predicted to score "
    text += f"**{prediction.get('predicted_points', 0):.1f} points** "
    
    if 'predicted_position' in prediction:
        text += f"with an expected finish of **P{int(prediction['predicted_position'])}**. "
    
    if 'confidence_80_lower' in prediction and 'confidence_80_upper' in prediction:
        text += f"\n\n*80% confidence interval: {prediction['confidence_80_lower']:.1f} - {prediction['confidence_80_upper']:.1f} points*"
    
    return text


def calculate_driver_statistics(driver_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive driver statistics"""
    if len(driver_data) == 0:
        return {}
    
    stats = {
        'total_races': len(driver_data),
        'total_points': driver_data['points'].sum() if 'points' in driver_data.columns else 0,
        'avg_position': driver_data['position'].mean() if 'position' in driver_data.columns else 0,
        'median_position': driver_data['position'].median() if 'position' in driver_data.columns else 0,
        'wins': len(driver_data[driver_data['position'] == 1]) if 'position' in driver_data.columns else 0,
        'podiums': len(driver_data[driver_data['position'] <= 3]) if 'position' in driver_data.columns else 0,
        'top_5': len(driver_data[driver_data['position'] <= 5]) if 'position' in driver_data.columns else 0,
        'points_per_race': driver_data['points'].mean() if 'points' in driver_data.columns else 0,
        'best_finish': driver_data['position'].min() if 'position' in driver_data.columns else 0,
        'worst_finish': driver_data['position'].max() if 'position' in driver_data.columns else 0
    }
    
    # DNFs
    if 'status' in driver_data.columns:
        stats['dnfs'] = len(driver_data[driver_data['status'].str.contains('Retired|Accident|Collision', case=False, na=False)])
    else:
        stats['dnfs'] = 0
    
    # Calculate consistency (std dev of positions)
    if 'position' in driver_data.columns and len(driver_data) > 1:
        stats['consistency'] = driver_data['position'].std()
    else:
        stats['consistency'] = 0
    
    # Win rate
    stats['win_rate'] = stats['wins'] / stats['total_races'] if stats['total_races'] > 0 else 0
    
    # Podium rate
    stats['podium_rate'] = stats['podiums'] / stats['total_races'] if stats['total_races'] > 0 else 0
    
    return stats


def get_recent_form(driver_data: pd.DataFrame, n_races: int = 5) -> Dict[str, float]:
    """Calculate recent form metrics"""
    if len(driver_data) == 0:
        return {}
    
    recent = driver_data.tail(n_races)
    
    form = {
        'avg_position': recent['position'].mean() if 'position' in recent.columns else 0,
        'avg_points': recent['points'].mean() if 'points' in recent.columns else 0,
        'total_points': recent['points'].sum() if 'points' in recent.columns else 0,
        'best_position': recent['position'].min() if 'position' in recent.columns else 0,
        'trend': 'unknown'
    }
    
    if 'position' in recent.columns and len(recent) >= 2:
        form['trend'] = 'improving' if recent['position'].iloc[-1] < recent['position'].iloc[0] else 'declining'
    
    return form


def format_time_delta(seconds: float) -> str:
    """Format time delta in seconds to readable format"""
    if pd.isna(seconds):
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.3f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"


def parse_lap_time(time_str: str) -> Optional[float]:
    """Parse lap time string to seconds"""
    try:
        if pd.isna(time_str) or time_str == '':
            return None
        
        # Format: "1:23.456"
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return None


def get_season_summary(data: pd.DataFrame, season: int) -> Dict:
    """Get summary statistics for a season"""
    season_data = data[data['season'] == season]
    
    summary = {
        'total_races': season_data['round'].nunique() if 'round' in season_data.columns else 0,
        'drivers': season_data['driver_id'].nunique() if 'driver_id' in season_data.columns else 0,
        'constructors': season_data['constructor_id'].nunique() if 'constructor_id' in season_data.columns else 0,
        'champion': None,
        'constructor_champion': None
    }
    
    # Find champions (most points)
    if len(season_data) > 0 and 'driver_id' in season_data.columns and 'points' in season_data.columns:
        driver_points = season_data.groupby('driver_id')['points'].sum().sort_values(ascending=False)
        if len(driver_points) > 0:
            summary['champion'] = driver_points.index[0]
        
        if 'constructor_id' in season_data.columns:
            constructor_points = season_data.groupby('constructor_id')['points'].sum().sort_values(ascending=False)
            if len(constructor_points) > 0:
                summary['constructor_champion'] = constructor_points.index[0]
    
    return summary


def cache_data(key: str, data: Any, cache_dir: Path = Path("data/cache")):
    """Cache data to disk"""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, f)
    except Exception as e:
        print(f"Error caching data: {e}")


def load_cached_data(key: str, max_age_hours: int = 24, cache_dir: Path = Path("data/cache")) -> Optional[Any]:
    """Load cached data if it exists and is fresh"""
    try:
        cache_file = cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        
        # Check if cache is still fresh
        cache_time = datetime.fromisoformat(cached['timestamp'])
        age = datetime.now() - cache_time
        
        if age.total_seconds() / 3600 > max_age_hours:
            return None
        
        return cached['data']
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return False
    return True


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def get_color_palette(theme: str = 'f1') -> Dict[str, str]:
    """Get color palette for visualizations"""
    palettes = {
        'f1': {
            'primary': '#E10600',
            'secondary': '#FF1E00',
            'accent': '#15151E',
            'success': '#00D962',
            'warning': '#FFB81C',
            'info': '#0090FF'
        },
        'dark': {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'accent': '#f093fb',
            'success': '#4ade80',
            'warning': '#fbbf24',
            'info': '#60a5fa'
        }
    }
    
    return palettes.get(theme, palettes['f1'])


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format float as percentage string"""
    return f"{value * 100:.{decimals}f}%"


def get_driver_color(constructor_id: str) -> str:
    """Get team color for a driver based on constructor"""
    team_colors = {
        'red_bull': '#0600EF',
        'ferrari': '#DC0000',
        'mercedes': '#00D2BE',
        'mclaren': '#FF8700',
        'alpine': '#0090FF',
        'aston_martin': '#006F62',
        'williams': '#005AFF',
        'alphatauri': '#2B4562',
        'alfa': '#900000',
        'haas': '#FFFFFF'
    }
    
    return team_colors.get(constructor_id, '#666666')


def create_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    return f'<a href="data:file/csv;base64,{csv}" download="{filename}">Download CSV</a>'


def format_delta(current: float, previous: float, better_lower: bool = True) -> str:
    """Format delta between two values with arrow indicator"""
    delta = current - previous
    
    if delta == 0:
        return "→ 0"
    
    is_better = (delta < 0 and better_lower) or (delta > 0 and not better_lower)
    arrow = "↑" if delta > 0 else "↓"
    color = "green" if is_better else "red"
    
    return f'<span style="color: {color}">{arrow} {abs(delta):.1f}</span>'


# Constants
CIRCUITS = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
    'Miami', 'Monaco', 'Spain', 'Canada', 'Austria',
    'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Singapore', 'USA', 'Mexico', 'Brazil', 'Abu Dhabi'
]

DRIVER_ABBREVIATIONS = {
    'max_verstappen': 'VER',
    'lewis_hamilton': 'HAM',
    'charles_leclerc': 'LEC',
    'lando_norris': 'NOR',
    'oscar_piastri': 'PIA'
}


if __name__ == "__main__":
    # Test functions
    config = load_config()
    print("Config loaded:", config)
    
    # Test with sample data
    sample_df = pd.DataFrame({
        'position': [1, 2, 3, 1, 5],
        'points': [25, 18, 15, 25, 10],
        'status': ['Finished', 'Finished', 'Finished', 'Finished', 'Retired']
    })
    
    stats = calculate_driver_statistics(sample_df)
    print("Stats:", stats)
