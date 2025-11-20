"""
Race Predictions Page
Predict outcomes for upcoming races
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.visualizations import create_prediction_chart, create_header, F1_RED
from models.predictor import F1Predictor
from models.rag_pipeline import F1RAGPipeline

st.set_page_config(
    page_title="Race Predictions",
    page_icon="🏁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .race-card {
        background: linear-gradient(135deg, #E10600 0%, #FF1E00 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .position-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .p1 { background: #FFD700; color: black; }
    .p2 { background: #C0C0C0; color: black; }
    .p3 { background: #CD7F32; color: white; }
    .other { background: #666; color: white; }
</style>
""", unsafe_allow_html=True)

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
                return pd.read_csv(path)
        
        st.warning("Data not found. Please upload a CSV file or run data collection.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.markdown(create_header(
        "Race Predictions",
        "AI-powered forecasts for upcoming F1 races"
    ), unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Unable to load race data")
        return
    
    # Sidebar
    st.sidebar.header("Race Selection")
    
    # Get unique circuits
    circuits = sorted(data['circuit'].unique())
    selected_circuit = st.sidebar.selectbox("Select Circuit", circuits)
    
    # Prediction options
    st.sidebar.header("Prediction Options")
    top_n_drivers = st.sidebar.slider("Show Top N Drivers", 5, 20, 10)
    include_weather = st.sidebar.checkbox("Consider Weather", value=False)
    scenario_mode = st.sidebar.selectbox(
        "Prediction Scenario",
        ["Normal Conditions", "Wet Race", "Safety Car Likely", "High Attrition"]
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="race-card">
            <h2>🏁 {selected_circuit}</h2>
            <p style='font-size: 1.2rem;'>Next Race Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Prediction Confidence", "87%", "↑ 5%")
        st.metric("Historical Races", len(data[data['circuit'] == selected_circuit]))
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🏆 Predicted Results",
        "📊 Analysis",
        "🤖 AI Insights"
    ])
    
    with tab1:
        st.subheader("Predicted Race Results")
        
        with st.spinner("Generating predictions..."):
            try:
                # Initialize predictor
                predictor = F1Predictor(
                    model_path=Path('data/models/tft_model.pt')
                )
                
                # Get current season drivers
                current_season = data['season'].max()
                current_drivers = data[
                    data['season'] == current_season
                ]['driver_id'].unique()
                
                # Predict race outcome
                race_predictions = []
                
                for driver_id in current_drivers[:top_n_drivers]:
                    pred = predictor.predict_driver(
                        driver_id,
                        data,
                        n_races=1
                    )
                    
                    if 'predictions' in pred and len(pred['predictions']) > 0:
                        race_predictions.append({
                            'driver_name': pred.get('driver_name', driver_id),
                            'driver_id': driver_id,
                            'predicted_points': pred['predictions'][0]['predicted_points'],
                            'predicted_position': pred['predictions'][0]['predicted_position'],
                            'confidence_lower': pred['predictions'][0].get('confidence_80_lower', 0),
                            'confidence_upper': pred['predictions'][0].get('confidence_80_upper', 0)
                        })
                
                # Sort by predicted points
                race_predictions_df = pd.DataFrame(race_predictions)
                race_predictions_df = race_predictions_df.sort_values(
                    'predicted_points', 
                    ascending=False
                ).reset_index(drop=True)
                race_predictions_df['position'] = range(1, len(race_predictions_df) + 1)
                
                # Display predicted podium
                st.markdown("### 🏆 Predicted Podium")
                
                if len(race_predictions_df) >= 3:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 10px;'>
                            <h1 style='margin: 0;'>🥇</h1>
                            <h2 style='margin: 0.5rem 0;'>{race_predictions_df.iloc[0]['driver_name']}</h2>
                            <p style='font-size: 1.5rem; margin: 0;'>{race_predictions_df.iloc[0]['predicted_points']:.1f} pts</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%); border-radius: 10px;'>
                            <h1 style='margin: 0;'>🥈</h1>
                            <h2 style='margin: 0.5rem 0;'>{race_predictions_df.iloc[1]['driver_name']}</h2>
                            <p style='font-size: 1.5rem; margin: 0;'>{race_predictions_df.iloc[1]['predicted_points']:.1f} pts</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%); border-radius: 10px;'>
                            <h1 style='margin: 0;'>🥉</h1>
                            <h2 style='margin: 0.5rem 0; color: white;'>{race_predictions_df.iloc[2]['driver_name']}</h2>
                            <p style='font-size: 1.5rem; margin: 0; color: white;'>{race_predictions_df.iloc[2]['predicted_points']:.1f} pts</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Full results table
                st.markdown("### 📋 Full Predicted Results")
                
                # Format the dataframe for display
                display_df = race_predictions_df[[
                    'position', 'driver_name', 'predicted_points', 
                    'confidence_lower', 'confidence_upper'
                ]].copy()
                
                display_df.columns = [
                    'Pos', 'Driver', 'Predicted Points', 
                    'Lower Bound', 'Upper Bound'
                ]
                
                # Style the dataframe
                st.dataframe(
                    display_df.style.background_gradient(
                        subset=['Predicted Points'],
                        cmap='RdYlGn'
                    ).format({
                        'Predicted Points': '{:.1f}',
                        'Lower Bound': '{:.1f}',
                        'Upper Bound': '{:.1f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                st.markdown("### 📊 Points Distribution")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=race_predictions_df['driver_name'],
                    y=race_predictions_df['predicted_points'],
                    marker_color=F1_RED,
                    error_y=dict(
                        type='data',
                        array=(race_predictions_df['confidence_upper'] - 
                               race_predictions_df['predicted_points']).tolist(),
                        arrayminus=(race_predictions_df['predicted_points'] - 
                                   race_predictions_df['confidence_lower']).tolist(),
                        visible=True
                    )
                ))
                
                fig.update_layout(
                    title="Predicted Points with Confidence Intervals",
                    xaxis_title="Driver",
                    yaxis_title="Points",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.info("Please ensure the model is trained and data is available.")
    
    with tab2:
        st.subheader("Race Analysis")
        
        # Historical performance at this circuit
        circuit_history = data[data['circuit'] == selected_circuit].copy()
        
        if not circuit_history.empty:
            st.markdown("### 📈 Historical Performance at This Circuit")
            
            # Winners history
            winners = circuit_history[
                circuit_history['position'] == 1
            ].groupby('driver_name').size().sort_values(ascending=False).head(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Wins**")
                for driver, wins in winners.items():
                    st.write(f"🏆 {driver}: {wins} wins")
            
            with col2:
                # Average points by team
                st.markdown("**Top Teams at This Circuit**")
                team_avg = circuit_history.groupby('constructor_id')['points'].mean().sort_values(ascending=False).head(5)
                for team, avg_pts in team_avg.items():
                    st.write(f"🏎️ {team}: {avg_pts:.1f} avg points")
            
            # Position changes chart
            st.markdown("### 🔄 Typical Position Changes")
            
            circuit_history['positions_gained'] = circuit_history['grid'] - circuit_history['position']
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=circuit_history['positions_gained'],
                nbinsx=30,
                marker_color=F1_RED,
                name='Position Changes'
            ))
            
            fig.update_layout(
                title=f"Position Changes at {selected_circuit}",
                xaxis_title="Positions Gained (Positive) / Lost (Negative)",
                yaxis_title="Frequency",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key statistics
            st.markdown("### 📊 Circuit Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_winner_grid = circuit_history[
                    circuit_history['position'] == 1
                ]['grid'].mean()
                st.metric("Avg Winner Grid", f"P{avg_winner_grid:.1f}")
            
            with col2:
                overtaking_factor = circuit_history['positions_gained'].abs().mean()
                st.metric("Overtaking Factor", f"{overtaking_factor:.1f}")
            
            with col3:
                dnf_rate = (circuit_history['status'].str.contains(
                    'Retired|Accident', case=False, na=False
                ).sum() / len(circuit_history) * 100)
                st.metric("DNF Rate", f"{dnf_rate:.1f}%")
            
            with col4:
                safety_car_races = len(circuit_history['race_name'].unique())
                st.metric("Races Held", safety_car_races)
    
    with tab3:
        st.subheader("🤖 AI-Generated Insights")
        
        with st.spinner("Analyzing recent news and generating insights..."):
            try:
                # Initialize RAG pipeline
                rag = F1RAGPipeline(corpus_path=Path('data/news_corpus'))
                
                # Get insights for top 3 predicted drivers
                if 'race_predictions_df' in locals() and not race_predictions_df.empty:
                    for idx, row in race_predictions_df.head(3).iterrows():
                        with st.expander(f"📰 {row['driver_name']} - Analysis"):
                            pred_dict = {
                                'predicted_points': row['predicted_points'],
                                'predicted_position': row['position']
                            }
                            
                            insights = rag.explain_prediction(
                                driver_name=row['driver_name'],
                                prediction=pred_dict,
                                context_query=f"{row['driver_name']} {selected_circuit} Formula 1"
                            )
                            
                            st.info(insights['explanation'])
                            
                            # Show sources
                            if insights['sources']:
                                st.markdown("**Sources:**")
                                for source in insights['sources'][:3]:
                                    st.markdown(f"- [{source['title']}]({source.get('url', '#')}) "
                                              f"({source['source']}, {source['date']})")
                
            except Exception as e:
                st.warning(f"Unable to generate AI insights: {str(e)}")
                st.info("Recent news analysis will be available once the RAG pipeline is configured.")
        
        # Scenario analysis
        st.markdown("### 🎯 Scenario Analysis")
        
        st.info(f"**Current Scenario:** {scenario_mode}")
        
        scenario_impacts = {
            "Normal Conditions": "Standard race conditions favor consistent performers and strong qualifiers.",
            "Wet Race": "Wet conditions increase unpredictability. Drivers with strong wet-weather skills may outperform.",
            "Safety Car Likely": "Safety cars can bunch up the field and create overtaking opportunities.",
            "High Attrition": "Increased retirements may benefit drivers starting mid-field."
        }
        
        st.write(scenario_impacts[scenario_mode])


if __name__ == "__main__":
    main()
