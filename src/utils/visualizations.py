"""
Visualization Utilities
Creates Plotly charts and graphs for F1 data
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# F1 Team Colors
TEAM_COLORS = {
    'red_bull': '#0600EF',
    'ferrari': '#DC0000',
    'mercedes': '#00D2BE',
    'mclaren': '#FF8700',
    'alpine': '#0090FF',
    'aston_martin': '#006F62',
    'williams': '#005AFF',
    'alphatauri': '#2B4562',
    'alfa': '#900000',
    'haas': '#FFFFFF',
    'racing_point': '#F596C8',
    'renault': '#FFF500'
}

F1_RED = '#E10600'


def create_performance_timeline(
    data: pd.DataFrame,
    metric: str = 'points',
    title: str = "Performance Timeline"
) -> go.Figure:
    """Create performance timeline chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['race_name'],
        y=data[metric],
        mode='lines+markers',
        name=metric.title(),
        line=dict(color=F1_RED, width=2),
        marker=dict(size=8, line=dict(width=2, color='white'))
    ))
    
    # Add rolling average if enough data
    if len(data) >= 5:
        data['rolling_avg'] = data[metric].rolling(window=5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=data['race_name'],
            y=data['rolling_avg'],
            mode='lines',
            name='5-Race Average',
            line=dict(color='rgba(225, 6, 0, 0.5)', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Race",
        yaxis_title=metric.title(),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_prediction_chart(
    predictions: List[Dict],
    historical_data: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create prediction chart with confidence intervals"""
    
    fig = go.Figure()
    
    races = [p['race_number'] for p in predictions]
    predicted_points = [p['predicted_points'] for p in predictions]
    
    # Historical data
    if historical_data is not None:
        fig.add_trace(go.Scatter(
            x=list(range(-(len(historical_data)-1), 1)),
            y=historical_data['points'].tolist(),
            mode='lines+markers',
            name='Historical',
            line=dict(color='gray', width=2),
            marker=dict(size=6)
        ))
    
    # Confidence intervals
    if 'confidence_95_lower' in predictions[0]:
        lower_95 = [p['confidence_95_lower'] for p in predictions]
        upper_95 = [p['confidence_95_upper'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=races + races[::-1],
            y=upper_95 + lower_95[::-1],
            fill='toself',
            fillcolor='rgba(225, 6, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
    
    if 'confidence_80_lower' in predictions[0]:
        lower_80 = [p['confidence_80_lower'] for p in predictions]
        upper_80 = [p['confidence_80_upper'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=races + races[::-1],
            y=upper_80 + lower_80[::-1],
            fill='toself',
            fillcolor='rgba(225, 6, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% Confidence',
            showlegend=True
        ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=races,
        y=predicted_points,
        mode='lines+markers',
        name='Predicted',
        line=dict(color=F1_RED, width=3),
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Performance Predictions",
        xaxis_title="Race Number",
        yaxis_title="Points",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_driver_comparison(
    drivers_data: Dict[str, pd.DataFrame],
    metric: str = 'points'
) -> go.Figure:
    """Compare multiple drivers"""
    
    fig = go.Figure()
    
    for driver_name, data in drivers_data.items():
        fig.add_trace(go.Scatter(
            x=data['race_name'],
            y=data[metric],
            mode='lines+markers',
            name=driver_name,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Driver Comparison - {metric.title()}",
        xaxis_title="Race",
        yaxis_title=metric.title(),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_team_performance_heatmap(
    data: pd.DataFrame,
    season: int
) -> go.Figure:
    """Create heatmap of team performance across races"""
    
    season_data = data[data['season'] == season]
    
    # Pivot table: teams vs races
    pivot = season_data.pivot_table(
        values='points',
        index='constructor_id',
        columns='race_name',
        aggfunc='sum'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 10},
        colorbar=dict(title="Points")
    ))
    
    fig.update_layout(
        title=f"Team Performance Heatmap - {season}",
        xaxis_title="Race",
        yaxis_title="Team",
        height=500
    )
    
    return fig


def create_position_distribution(
    data: pd.DataFrame,
    driver_name: str
) -> go.Figure:
    """Create histogram of finishing positions"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data['position'],
        nbinsx=20,
        marker_color=F1_RED,
        name='Positions'
    ))
    
    fig.update_layout(
        title=f"Finishing Position Distribution - {driver_name}",
        xaxis_title="Position",
        yaxis_title="Frequency",
        showlegend=False,
        template='plotly_white',
        height=400
    )
    
    return fig


def create_points_progression(
    data: pd.DataFrame,
    season: int,
    top_n: int = 5
) -> go.Figure:
    """Create cumulative points progression chart"""
    
    season_data = data[data['season'] == season].copy()
    season_data = season_data.sort_values(['round', 'driver_id'])
    
    # Calculate cumulative points
    season_data['cumulative_points'] = season_data.groupby('driver_id')['points'].cumsum()
    
    # Get top N drivers by final points
    final_standings = season_data.groupby('driver_id')['cumulative_points'].max().nlargest(top_n)
    
    fig = go.Figure()
    
    for driver_id in final_standings.index:
        driver_data = season_data[season_data['driver_id'] == driver_id]
        
        fig.add_trace(go.Scatter(
            x=driver_data['race_name'],
            y=driver_data['cumulative_points'],
            mode='lines+markers',
            name=driver_data['driver_name'].iloc[0] if 'driver_name' in driver_data.columns else driver_id,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Championship Progression - {season}",
        xaxis_title="Race",
        yaxis_title="Cumulative Points",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_qualifying_race_correlation(
    data: pd.DataFrame
) -> go.Figure:
    """Scatter plot: grid position vs finish position"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['grid'],
        y=data['position'],
        mode='markers',
        marker=dict(
            size=8,
            color=data['points'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Points"),
            line=dict(width=1, color='white')
        ),
        text=data['driver_name'] if 'driver_name' in data.columns else None,
        hovertemplate='Grid: %{x}<br>Finish: %{y}<br>%{text}<extra></extra>'
    ))
    
    # Add diagonal line (grid = finish)
    max_pos = max(data['grid'].max(), data['position'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_pos],
        y=[0, max_pos],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Grid = Finish',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Qualifying vs Race Position",
        xaxis_title="Grid Position",
        yaxis_title="Finish Position",
        template='plotly_white',
        height=500
    )
    
    return fig


def create_race_pace_chart(
    lap_times: pd.DataFrame,
    drivers: List[str]
) -> go.Figure:
    """Create race pace comparison chart"""
    
    fig = go.Figure()
    
    for driver in drivers:
        driver_laps = lap_times[lap_times['driver_id'] == driver]
        
        fig.add_trace(go.Scatter(
            x=driver_laps['lap'],
            y=driver_laps['time_seconds'],
            mode='lines',
            name=driver,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Race Pace Comparison",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_gauge_chart(
    value: float,
    max_value: float = 26,
    title: str = "Predicted Points"
) -> go.Figure:
    """Create gauge chart for single metric"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': F1_RED},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "lightgray"},
                {'range': [max_value*0.33, max_value*0.66], 'color': "gray"},
                {'range': [max_value*0.66, max_value], 'color': "#FFD700"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    
    fig.update_layout(height=250)
    
    return fig


def create_radar_chart(
    driver_stats: Dict[str, float],
    title: str = "Driver Performance Radar"
) -> go.Figure:
    """Create radar chart for multi-dimensional comparison"""
    
    categories = list(driver_stats.keys())
    values = list(driver_stats.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(225, 6, 0, 0.3)',
        line=dict(color=F1_RED, width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig


def create_confidence_interval_chart(
    predictions: pd.DataFrame
) -> go.Figure:
    """Create confidence interval visualization"""
    
    fig = go.Figure()
    
    x = predictions['race_number']
    
    # 95% CI
    fig.add_trace(go.Scatter(
        x=x,
        y=predictions['confidence_95_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=predictions['confidence_95_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(225, 6, 0, 0.1)',
        name='95% CI',
        hoverinfo='skip'
    ))
    
    # 80% CI
    fig.add_trace(go.Scatter(
        x=x,
        y=predictions['confidence_80_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=predictions['confidence_80_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(225, 6, 0, 0.2)',
        name='80% CI',
        hoverinfo='skip'
    ))
    
    # Median prediction
    fig.add_trace(go.Scatter(
        x=x,
        y=predictions['predicted_points'],
        mode='lines+markers',
        name='Prediction',
        line=dict(color=F1_RED, width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Prediction with Confidence Intervals",
        xaxis_title="Race Number",
        yaxis_title="Points",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_header(
    title: str,
    subtitle: str = "",
    icon: str = "🏎️"
) -> str:
    """Create styled header HTML"""
    
    return f"""
    <div style='background: linear-gradient(90deg, #E10600 0%, #FF1E00 100%); 
                padding: 2rem; 
                border-radius: 10px; 
                color: white; 
                margin-bottom: 2rem;'>
        <h1 style='margin: 0;'>{icon} {title}</h1>
        {f"<p style='font-size: 1.2rem; margin-top: 0.5rem;'>{subtitle}</p>" if subtitle else ""}
    </div>
    """


# Export all functions
__all__ = [
    'create_performance_timeline',
    'create_prediction_chart',
    'create_driver_comparison',
    'create_team_performance_heatmap',
    'create_position_distribution',
    'create_points_progression',
    'create_qualifying_race_correlation',
    'create_race_pace_chart',
    'create_gauge_chart',
    'create_radar_chart',
    'create_confidence_interval_chart',
    'create_header',
    'TEAM_COLORS',
    'F1_RED'
]
