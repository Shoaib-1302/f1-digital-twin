"""
Team Comparison Page
Compare constructor performance and strategies
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from _imports import *

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualizations import (
    create_team_performance_heatmap,
    create_points_progression,
    create_header,
    TEAM_COLORS
)

st.set_page_config(
    page_title="Team Comparison",
    page_icon="📊",
    layout="wide"
)

st.markdown(create_header(
    "Team Comparison",
    "Constructor performance analysis and head-to-head comparisons"
), unsafe_allow_html=True)


def load_data():
    try:
        possible_paths = [
            'data/processed/processed_data.csv',
            'data/raw/results.csv',
            f'{BASE_PATH}/data/processed/processed_data.csv',
            f'{BASE_PATH}/data/raw/results.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                if 'team_data' not in st.session_state:
                    st.session_state.team_data = df
                return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main():
    data = load_data()
    
    if data is None:
        st.error("Unable to load data")
        return
    
    # Sidebar
    st.sidebar.header("Team Selection")
    
    # Get unique teams
    teams = sorted(data['constructor_id'].unique())
    
    # Multi-select for comparison
    selected_teams = st.sidebar.multiselect(
        "Select Teams to Compare",
        teams,
        default=teams[:3] if len(teams) >= 3 else teams
    )
    
    # Season selection
    seasons = sorted(data['season'].unique(), reverse=True)
    selected_season = st.sidebar.selectbox("Season", seasons)
    
    if not selected_teams:
        st.warning("Please select at least one team from the sidebar")
        return
    
    # Filter data
    season_data = data[data['season'] == selected_season]
    team_data = season_data[season_data['constructor_id'].isin(selected_teams)]
    
    # Team overview metrics
    st.header(f"📊 {selected_season} Season Overview")
    
    cols = st.columns(len(selected_teams))
    
    for idx, team in enumerate(selected_teams):
        team_season = team_data[team_data['constructor_id'] == team]
        
        with cols[idx]:
            total_points = team_season['points'].sum()
            wins = len(team_season[team_season['position'] == 1])
            podiums = len(team_season[team_season['position'] <= 3])
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <h3 style='margin: 0;'>{team.replace('_', ' ').title()}</h3>
                <h1 style='margin: 0.5rem 0;'>{total_points:.0f}</h1>
                <p style='margin: 0;'>Points</p>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p style='margin: 0.5rem 0;'>🏆 {wins} Wins | 🥇 {podiums} Podiums</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Trends",
        "👥 Driver Comparison",
        "🏁 Race-by-Race",
        "📊 Statistics"
    ])
    
    with tab1:
        st.subheader("Performance Trends")
        
        # Points progression
        fig_progression = create_points_progression(team_data, selected_season, top_n=len(selected_teams))
        st.plotly_chart(fig_progression, use_container_width=True)
        
        # Average position over season
        st.markdown("### Average Finish Position")
        
        avg_position = team_data.groupby(['constructor_id', 'round']).agg({
            'position': 'mean',
            'points': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        for team in selected_teams:
            team_avg = avg_position[avg_position['constructor_id'] == team]
            
            fig.add_trace(go.Scatter(
                x=team_avg['round'],
                y=team_avg['position'],
                mode='lines+markers',
                name=team.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Average Position",
            yaxis=dict(autorange='reversed'),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Points per race
        st.markdown("### Points Per Race")
        
        points_per_race = team_data.groupby(['constructor_id', 'race_name'])['points'].sum().reset_index()
        
        fig = px.box(
            points_per_race,
            x='constructor_id',
            y='points',
            color='constructor_id',
            title="Points Distribution by Team"
        )
        
        fig.update_layout(
            xaxis_title="Team",
            yaxis_title="Points",
            showlegend=False,
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Driver Comparison Within Teams")
        
        for team in selected_teams:
            team_drivers = team_data[team_data['constructor_id'] == team]
            drivers = team_drivers['driver_name'].unique()
            
            if len(drivers) >= 2:
                with st.expander(f"🏎️ {team.replace('_', ' ').title()}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    for idx, driver in enumerate(drivers[:2]):
                        driver_data = team_drivers[team_drivers['driver_name'] == driver]
                        
                        with col1 if idx == 0 else col2:
                            st.markdown(f"#### {driver}")
                            
                            total_points = driver_data['points'].sum()
                            avg_pos = driver_data['position'].mean()
                            best_pos = driver_data['position'].min()
                            
                            st.metric("Total Points", f"{total_points:.0f}")
                            st.metric("Avg Position", f"P{avg_pos:.1f}")
                            st.metric("Best Finish", f"P{int(best_pos)}")
                    
                    # Head-to-head chart
                    driver_comparison = team_drivers.groupby('driver_name').agg({
                        'points': 'sum',
                        'position': 'mean'
                    }).reset_index()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=driver_comparison['driver_name'],
                        y=driver_comparison['points'],
                        name='Total Points',
                        marker_color='#E10600'
                    ))
                    
                    fig.update_layout(
                        title=f"{team.replace('_', ' ').title()} - Driver Comparison",
                        xaxis_title="Driver",
                        yaxis_title="Total Points",
                        template='plotly_white',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Race-by-Race Performance")
        
        # Performance heatmap
        if len(selected_teams) > 1:
            fig_heatmap = create_team_performance_heatmap(team_data, selected_season)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Detailed race results
        st.markdown("### Detailed Results")
        
        # Create pivot table
        race_results = team_data.pivot_table(
            values='points',
            index='race_name',
            columns='constructor_id',
            aggfunc='sum',
            fill_value=0
        )
        
        st.dataframe(
            race_results.style.background_gradient(cmap='RdYlGn', axis=1).format("{:.0f}"),
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Statistical Analysis")
        
        # Calculate statistics for each team
        team_stats = []
        
        for team in selected_teams:
            team_season = team_data[team_data['constructor_id'] == team]
            
            stats = {
                'Team': team.replace('_', ' ').title(),
                'Total Points': team_season['points'].sum(),
                'Avg Points/Race': team_season.groupby('race_name')['points'].sum().mean(),
                'Wins': len(team_season[team_season['position'] == 1]),
                'Podiums': len(team_season[team_season['position'] <= 3]),
                'Top 5s': len(team_season[team_season['position'] <= 5]),
                'DNFs': len(team_season[team_season['status'].str.contains('Retired|Accident', case=False, na=False)]),
                'Avg Position': team_season['position'].mean(),
                'Best Position': team_season['position'].min(),
                'Consistency (σ)': team_season['position'].std()
            }
            
            team_stats.append(stats)
        
        stats_df = pd.DataFrame(team_stats)
        
        # Display statistics
        st.dataframe(
            stats_df.style.highlight_max(
                subset=['Total Points', 'Wins', 'Podiums'],
                color='lightgreen'
            ).highlight_min(
                subset=['Avg Position', 'DNFs', 'Consistency (σ)'],
                color='lightgreen'
            ).format({
                'Total Points': '{:.0f}',
                'Avg Points/Race': '{:.1f}',
                'Avg Position': '{:.2f}',
                'Best Position': '{:.0f}',
                'Consistency (σ)': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Radar chart comparison
        if len(selected_teams) == 2:
            st.markdown("### Head-to-Head Radar Comparison")
            
            team1_data = stats_df.iloc[0]
            team2_data = stats_df.iloc[1]
            
            categories = ['Wins', 'Podiums', 'Top 5s', 'Avg Points/Race', 'Consistency (σ)']
            
            # Normalize values
            team1_values = [
                team1_data['Wins'] / max(stats_df['Wins'].max(), 1) * 100,
                team1_data['Podiums'] / max(stats_df['Podiums'].max(), 1) * 100,
                team1_data['Top 5s'] / max(stats_df['Top 5s'].max(), 1) * 100,
                team1_data['Avg Points/Race'] / max(stats_df['Avg Points/Race'].max(), 1) * 100,
                100 - (team1_data['Consistency (σ)'] / max(stats_df['Consistency (σ)'].max(), 1) * 100)
            ]
            
            team2_values = [
                team2_data['Wins'] / max(stats_df['Wins'].max(), 1) * 100,
                team2_data['Podiums'] / max(stats_df['Podiums'].max(), 1) * 100,
                team2_data['Top 5s'] / max(stats_df['Top 5s'].max(), 1) * 100,
                team2_data['Avg Points/Race'] / max(stats_df['Avg Points/Race'].max(), 1) * 100,
                100 - (team2_data['Consistency (σ)'] / max(stats_df['Consistency (σ)'].max(), 1) * 100)
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=team1_values,
                theta=categories,
                fill='toself',
                name=team1_data['Team'],
                line=dict(color='#E10600', width=2)
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=team2_values,
                theta=categories,
                fill='toself',
                name=team2_data['Team'],
                line=dict(color='#0600EF', width=2)
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
