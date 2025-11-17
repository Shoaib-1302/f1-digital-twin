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
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict:
    """Return default configuration"""
    return {
        'data': {
            'start_year': 2018,
            'end_year': 2024,
            'min_races_for_prediction': 5
        },
        'visualization': {
            'theme': 'plotly_dark',
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
            'ttl_seconds': 3600,
            'max_size_mb': 1000
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
    stats = {
        'total_races': len(driver_data),
        'total_points': driver_data['points'].sum(),
        'avg_position': driver_data['position'].mean(),
        'median_position': driver_data['position'].median(),
        'wins': len(driver_data[driver_data['position'] == 1]),
        'podiums': len(driver_data[driver_data['position'] <= 3]),
        'top_5': len(driver_data[driver_data['position'] <= 5]),
        'dnfs': len(driver_data[driver_data['status'].str.contains('Retired|Accident|Collision', case=False, na=False)]),
        'points_per_race': driver_data['points'].mean(),
        'best_finish': driver_data['position'].min(),
        'worst_finish': driver_data['position'].max()
    }
    
    # Calculate consistency (std dev of positions)
    stats['consistency'] = driver_data['position'].std()
    
    # Win rate
    stats['win_rate'] = stats['wins'] / stats['total_races'] if stats['total_races'] > 0 else 0
    
    # Podium rate
    stats['podium_rate'] = stats['podiums'] / stats['total_races'] if stats['total_races'] > 0 else 0
    
    return stats


def get_recent_form(driver_data: pd.DataFrame, n_races: int = 5) -> Dict[str, float]:
    """Calculate recent form metrics"""
    recent = driver_data.tail(n_races)
    
    form = {
        'avg_position': recent['position'].mean(),
        'avg_points': recent['points'].mean(),
        'total_points': recent['points'].sum(),
        'best_position': recent['position'].min(),
        'trend': 'improving' if recent['position'].iloc[-1] < recent['position'].iloc[0] else 'declining'
    }
    
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
        'total_races': season_data['round'].nunique(),
        'drivers': season_data['driver_id'].nunique(),
        'constructors': season_data['constructor_id'].nunique(),
        'champion': None,
        'constructor_champion': None
    }
    
    # Find champions (most points)
    if len(season_data) > 0:
        driver_points = season_data.groupby('driver_id')['points'].sum().sort_values(ascending=False)
        if len(driver_points) > 0:
            summary['champion'] = driver_points.index[0]
        
        constructor_points = season_data.groupby('constructor_id')['points'].sum().sort_values(ascending=False)
        if len(constructor_points) > 0:
            summary['constructor_champion'] = constructor_points.index[0]
    
    return summary


def cache_data(key: str, data: Any, cache_dir: Path = Path("data/cache")):
    """Cache data to disk"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'data': data,
                'timestamp': datetime.now().isoformat()
            }, f)
    except Exception as e:
        print(f"Error caching data: {e}")


def load_cached_data(key: str, max_age_hours: int = 24, cache_dir: Path = Path("data/cache")) -> Optional[Any]:
    """Load cached data if it exists and is fresh"""
    cache_file = cache_dir / f"{key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
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
    # Add more as needed
}


if __name__ == "__main__":
    # Test functions
    config = load_config()
    print("Config loaded:", config)
    
    stats = calculate_driver_statistics(pd.DataFrame({
        'position': [1, 2, 3, 1, 5],
        'points': [25, 18, 15, 25, 10],
        'status': ['Finished', 'Finished', 'Finished', 'Finished', 'Retired']
    }))
    print("Stats:", stats)
