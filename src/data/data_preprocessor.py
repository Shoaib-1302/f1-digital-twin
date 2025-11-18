"""
Data Preprocessing and Feature Engineering Module
Transforms raw F1 data into model-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class F1DataPreprocessor:
    """Preprocesses F1 data and engineers features for modeling"""
    
    def __init__(self):
        self.feature_columns = []
        self.target_columns = ['points', 'position']
        
    def process_race_results(
        self,
        results_df: pd.DataFrame,
        include_rolling_stats: bool = True,
        lookback_races: int = 5
    ) -> pd.DataFrame:
        """
        Process raw race results and create features
        
        Args:
            results_df: Raw race results DataFrame
            include_rolling_stats: Whether to calculate rolling statistics
            lookback_races: Number of races for rolling calculations
        """
        
        df = results_df.copy()
        
        # Convert data types
        df['season'] = df['season'].astype(int)
        df['round'] = df['round'].astype(int)
        df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        
        # Create unique race index
        df = df.sort_values(['season', 'round', 'driver_id'])
        df['race_index'] = df.groupby('driver_id').cumcount()
        
        # Driver experience features
        df = self._add_driver_experience_features(df)
        
        # Rolling performance features
        if include_rolling_stats:
            df = self._add_rolling_features(df, lookback_races)
        
        # Constructor features
        df = self._add_constructor_features(df)
        
        # Circuit features
        df = self._add_circuit_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Position change features
        df = self._add_position_change_features(df)
        
        # Form and momentum features
        df = self._add_form_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _add_driver_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add driver experience and career statistics"""
        
        # Career races
        df['career_races'] = df.groupby('driver_id').cumcount()
        
        # Career wins (cumulative)
        df['career_wins'] = (df['position'] == 1).astype(int)
        df['career_wins'] = df.groupby('driver_id')['career_wins'].cumsum()
        
        # Career podiums
        df['career_podiums'] = (df['position'] <= 3).astype(int)
        df['career_podiums'] = df.groupby('driver_id')['career_podiums'].cumsum()
        
        # Career points
        df['career_points'] = df.groupby('driver_id')['points'].cumsum()
        
        # Years in F1
        df['first_race_date'] = df.groupby('driver_id')['date'].transform('min')
        df['driver_experience_years'] = (df['date'] - df['first_race_date']).dt.days / 365.25
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add rolling average statistics"""
        
        # Sort by driver and race
        df = df.sort_values(['driver_id', 'race_index'])
        
        # Rolling average position
        df['rolling_avg_position_5'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df['rolling_avg_position_10'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # Rolling average points
        df['rolling_avg_points_5'] = df.groupby('driver_id')['points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df['rolling_avg_points_10'] = df.groupby('driver_id')['points'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # Rolling standard deviation (consistency)
        df['rolling_std_position_5'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # Rolling best position
        df['rolling_best_position_5'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        
        return df
    
    def _add_constructor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add constructor/team performance features"""
        
        # Constructor average points per race (in this season)
        constructor_avg = df.groupby(['season', 'constructor_id', 'round'])['points'].mean().reset_index()
        constructor_avg.columns = ['season', 'constructor_id', 'round', 'constructor_avg_points']
        df = df.merge(constructor_avg, on=['season', 'constructor_id', 'round'], how='left')
        
        # Constructor rolling performance
        df = df.sort_values(['constructor_id', 'season', 'round'])
        df['constructor_rolling_points_5'] = df.groupby(['constructor_id', 'season'])['points'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        return df
    
    def _add_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circuit-specific features"""
        
        # Encode circuit as numeric
        df['circuit_encoded'] = df.groupby('circuit').ngroup()
        
        # Driver's historical performance at this circuit
        driver_circuit_stats = df.groupby(['driver_id', 'circuit']).agg({
            'position': ['mean', 'min'],
            'points': 'mean'
        }).reset_index()
        
        driver_circuit_stats.columns = ['driver_id', 'circuit', 'driver_circuit_avg_pos', 
                                        'driver_circuit_best_pos', 'driver_circuit_avg_points']
        
        df = df.merge(driver_circuit_stats, on=['driver_id', 'circuit'], how='left')
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Days since last race
        df = df.sort_values(['driver_id', 'date'])
        df['days_since_last_race'] = df.groupby('driver_id')['date'].diff().dt.days
        df['days_since_last_race'] = df['days_since_last_race'].fillna(14)  # Default 2 weeks
        
        # Week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Month
        df['month'] = df['date'].dt.month
        
        return df
    
    def _add_position_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features related to position changes"""
        
        # Grid position vs finish position
        df['positions_gained'] = df['grid'] - df['position']
        df['positions_gained'] = df['positions_gained'].fillna(0)
        
        # Previous race position
        df = df.sort_values(['driver_id', 'race_index'])
        df['prev_race_position'] = df.groupby('driver_id')['position'].shift(1)
        
        # Position change from previous race
        df['position_change_from_prev'] = df['prev_race_position'] - df['position']
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form and momentum features"""
        
        df = df.sort_values(['driver_id', 'race_index'])
        
        # Points in last 3 races
        df['form_last_3_races'] = df.groupby('driver_id')['points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        )
        
        # Winning streak
        df['is_win'] = (df['position'] == 1).astype(int)
        df['winning_streak'] = df.groupby('driver_id')['is_win'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        
        # Podium streak
        df['is_podium'] = (df['position'] <= 3).astype(int)
        df['podium_streak'] = df.groupby('driver_id')['is_podium'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        
        # DNF indicator
        df['is_dnf'] = df['status'].str.contains('Retired|Accident|Collision', case=False, na=False).astype(int)
        df['races_since_dnf'] = df.groupby('driver_id')['is_dnf'].transform(
            lambda x: x[::-1].cumsum()[::-1]
        )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df
    
    def process_qualifying_data(self, qual_df: pd.DataFrame) -> pd.DataFrame:
        """Process qualifying data"""
        
        df = qual_df.copy()
        
        # Convert qualifying times to seconds
        df['q1_seconds'] = df['q1'].apply(self._parse_lap_time)
        df['q2_seconds'] = df['q2'].apply(self._parse_lap_time)
        df['q3_seconds'] = df['q3'].apply(self._parse_lap_time)
        
        # Best qualifying time
        df['best_qual_time'] = df[['q1_seconds', 'q2_seconds', 'q3_seconds']].min(axis=1)
        
        # Gap to pole position
        pole_times = df.groupby(['season', 'round'])['best_qual_time'].min().reset_index()
        pole_times.columns = ['season', 'round', 'pole_time']
        df = df.merge(pole_times, on=['season', 'round'], how='left')
        df['gap_to_pole'] = df['best_qual_time'] - df['pole_time']
        
        return df
    
    def _parse_lap_time(self, time_str: str) -> Optional[float]:
        """Parse lap time string to seconds"""
        try:
            if pd.isna(time_str) or time_str == '':
                return None
            
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return None
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        time_based: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits
        
        Args:
            df: Processed DataFrame
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            time_based: If True, split by time; otherwise random
        """
        
        if time_based:
            # Sort by date
            df = df.sort_values('date')
            
            n = len(df)
            train_end = int(n * (1 - test_size - validation_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
        else:
            # Random split (stratified by driver)
            from sklearn.model_selection import train_test_split
            
            train_val_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df['driver_id']
            )
            
            train_df, val_df = train_test_split(
                train_val_df, 
                test_size=validation_size/(1-test_size), 
                random_state=42,
                stratify=train_val_df['driver_id']
            )
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, df: pd.DataFrame, output_dir: Path):
        """Save processed data to disk"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        df.to_csv(output_dir / 'processed_data.csv', index=False)
        
        # Save train/val/test splits
        train_df, val_df, test_df = self.create_train_test_split(df)
        
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        print(f"Processed data saved to {output_dir}")
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")


def main():
    """Example usage"""
    from data_collector import F1DataCollector
    
    # Collect data
    collector = F1DataCollector()
    data = collector.collect_multi_season_data(2020, 2024)
    
    # Preprocess
    preprocessor = F1DataPreprocessor()
    processed_df = preprocessor.process_race_results(
        data['results'],
        include_rolling_stats=True,
        lookback_races=5
    )
    
    # Save
    preprocessor.save_processed_data(processed_df, Path('data/processed'))


if __name__ == "__main__":
    main()
