"""
Temporal Fusion Transformer Model for F1 Performance Prediction
Implements multi-horizon forecasting with attention mechanisms
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class F1TFTModel(pl.LightningModule):
    """
    Temporal Fusion Transformer for F1 driver performance prediction
    Predicts future race positions and points based on historical performance
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model will be initialized after seeing the data
        self.model = None
        self.training_data = None
        
    def prepare_dataset(
        self,
        data: pd.DataFrame,
        max_prediction_length: int = 3,
        max_encoder_length: int = 10,
        target: str = "points"
    ) -> TimeSeriesDataSet:
        """
        Prepare PyTorch Forecasting TimeSeriesDataSet
        
        Args:
            data: DataFrame with columns [driver_id, race_index, features...]
            max_prediction_length: How many races ahead to predict
            max_encoder_length: How many historical races to use
            target: Target variable to predict
        """
        
        # Ensure data is sorted
        data = data.sort_values(['driver_id', 'race_index'])
        
        # Define time-varying features (change over time)
        time_varying_known_reals = [
            'circuit_encoded',
            'weather_condition',
            'days_since_last_race'
        ]
        
        time_varying_unknown_reals = [
            'position',
            'grid_position',
            'points',
            'fastest_lap_rank',
            'rolling_avg_position_5',
            'rolling_avg_points_5',
            'form_last_3_races'
        ]
        
        # Static features (don't change for a driver)
        static_categoricals = ['driver_id', 'constructor_id']
        
        static_reals = [
            'driver_experience_years',
            'career_wins',
            'career_podiums'
        ]
        
        # Create the dataset
        training = TimeSeriesDataSet(
            data,
            time_idx='race_index',
            target=target,
            group_ids=['driver_id'],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=['driver_id'],
                transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        return training
    
    def initialize_model(self, training_data: TimeSeriesDataSet):
        """Initialize the TFT model with dataset statistics"""
        
        self.training_data = training_data
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=self.config.get('learning_rate', 0.001),
            hidden_size=self.config.get('hidden_size', 160),
            attention_head_size=self.config.get('attention_heads', 4),
            dropout=self.config.get('dropout', 0.1),
            hidden_continuous_size=self.config.get('hidden_continuous_size', 16),
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
    
    def predict_driver_performance(
        self,
        driver_data: pd.DataFrame,
        n_races: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Predict driver performance for next N races
        
        Args:
            driver_data: Historical data for specific driver
            n_races: Number of races to predict
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.model.eval()
        
        # Prepare prediction data
        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training_data,
            driver_data,
            predict=True,
            stop_randomization=True
        )
        
        pred_dataloader = pred_dataset.to_dataloader(
            train=False,
            batch_size=1,
            num_workers=0
        )
        
        # Make predictions
        with torch.no_grad():
            raw_predictions = self.model.predict(
                pred_dataloader,
                mode='raw',
                return_x=True
            )
        
        # Extract predictions (median, upper and lower quantiles)
        predictions = {
            'median': raw_predictions.output['prediction'][:, :, 3].numpy(),  # 50th percentile
            'upper_80': raw_predictions.output['prediction'][:, :, 5].numpy(),  # 90th percentile
            'lower_80': raw_predictions.output['prediction'][:, :, 1].numpy(),  # 10th percentile
            'upper_95': raw_predictions.output['prediction'][:, :, 6].numpy(),  # 97.5th percentile
            'lower_95': raw_predictions.output['prediction'][:, :, 0].numpy(),  # 2.5th percentile
        }
        
        return predictions
    
    def get_attention_weights(self, batch) -> np.ndarray:
        """
        Extract attention weights for interpretability
        Shows which time steps the model focuses on
        """
        
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        self.model.eval()
        
        with torch.no_grad():
            interpretation = self.model.interpret_output(
                batch,
                reduction='sum'
            )
        
        return interpretation['attention'].numpy()
    
    def get_variable_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        Shows which variables matter most for predictions
        """
        
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        importance = {}
        
        # Variable selection weights
        encoder_importance = self.model.encoder_variables_selection.weight.data.numpy()
        decoder_importance = self.model.decoder_variables_selection.weight.data.numpy()
        
        # Get variable names from dataset
        var_names = (
            self.training_data.reals +
            self.training_data.categoricals
        )
        
        for i, name in enumerate(var_names):
            importance[name] = {
                'encoder': float(encoder_importance[i]),
                'decoder': float(decoder_importance[i]) if i < len(decoder_importance) else 0
            }
        
        return importance


class F1PerformancePredictor:
    """High-level interface for making F1 predictions"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.scaler = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        config: Dict,
        max_epochs: int = 100
    ):
        """Train the TFT model"""
        
        # Initialize model
        self.model = F1TFTModel(config)
        
        # Prepare datasets
        train_dataset = self.model.prepare_dataset(train_data)
        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            val_data,
            predict=False,
            stop_randomization=True
        )
        
        # Create dataloaders
        train_dataloader = train_dataset.to_dataloader(
            train=True,
            batch_size=config.get('batch_size', 64),
            num_workers=4
        )
        
        val_dataloader = val_dataset.to_dataloader(
            train=False,
            batch_size=config.get('batch_size', 64),
            num_workers=4
        )
        
        # Initialize model with data
        self.model.initialize_model(train_dataset)
        
        # Train
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            gradient_clip_val=0.1,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min'
                ),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min'
                )
            ]
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    
    def predict_next_races(
        self,
        driver_id: str,
        historical_data: pd.DataFrame,
        n_races: int = 3
    ) -> pd.DataFrame:
        """
        Predict performance for next N races
        
        Returns DataFrame with predictions and confidence intervals
        """
        
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Filter data for specific driver
        driver_data = historical_data[
            historical_data['driver_id'] == driver_id
        ].copy()
        
        # Make predictions
        predictions = self.model.predict_driver_performance(
            driver_data,
            n_races=n_races
        )
        
        # Format results
        results = []
        for i in range(n_races):
            results.append({
                'race_number': i + 1,
                'predicted_points': predictions['median'][0, i],
                'confidence_80_lower': predictions['lower_80'][0, i],
                'confidence_80_upper': predictions['upper_80'][0, i],
                'confidence_95_lower': predictions['lower_95'][0, i],
                'confidence_95_upper': predictions['upper_95'][0, i]
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, path: Path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'training_data': self.model.training_data
        }, path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        checkpoint = torch.load(path)
        
        self.model = F1TFTModel(checkpoint['config'])
        self.model.training_data = checkpoint['training_data']
        self.model.initialize_model(checkpoint['training_data'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


if __name__ == "__main__":
    # Example usage
    config = {
        'hidden_size': 160,
        'attention_heads': 4,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    # Load your data
    # train_data = pd.read_csv('data/processed/train.csv')
    # val_data = pd.read_csv('data/processed/val.csv')
    
    # predictor = F1PerformancePredictor()
    # predictor.train(train_data, val_data, config, max_epochs=100)
    # predictor.save_model(Path('data/models/tft_model.pt'))
