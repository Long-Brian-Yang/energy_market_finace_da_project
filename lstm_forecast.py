import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# For statistical tests
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro, normaltest

# For additional plots
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
import shap

class JPEXPriceForecastLSTM:
    def __init__(self, output_dir='outputs_LSTM'):
        """Initialize the forecaster."""
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.output_dir = output_dir
        self.create_output_directories()
        self.sequence_length = 48 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_losses = []
        self.validation_losses = []
        
    def create_output_directories(self):
        """Create base output directories."""
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_data(self, file_path):
        """Load and preprocess JPEX data."""
        # Read data
        df = pd.read_csv(file_path, encoding='shift-jis')
        
        columns = {
            '日付': 'date',
            '受渡日': 'date',  
            '時刻コード': 'time_code',
            '売り入札量(kWh)': 'sell_volume',
            '買い入札量(kWh)': 'buy_volume',
            '約定総量(kWh)': 'trading_volume',
            'システムプライス(円/kWh)': 'system_price'
        }
        df = df.rename(columns=columns)
        
        df['date'] = '2024-01-01' 
        df['date'] = pd.to_datetime(df['date'])
        print("Added 'date' column manually.")
        
        # Create time features
        df['hour'] = (df['time_code'] - 1) // 2
        df['minute'] = ((df['time_code'] - 1) % 2) * 30

        # Create 'datetime' column
        df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='m')
        
        # Calculate supply-demand imbalance
        df['volume_imbalance'] = df['buy_volume'] - df['sell_volume']
        df['volume_ratio'] = df['buy_volume'] / df['sell_volume']
        
        return df
    
    def create_features(self, df):
        """Create features."""
        # Lag features for price and trading volume
        for i in [1, 2, 48]:  # 30 minutes, 1 hour, 24 hours
            df[f'price_lag_{i}'] = df['system_price'].shift(i)
            df[f'volume_lag_{i}'] = df['trading_volume'].shift(i)
            df[f'imbalance_lag_{i}'] = df['volume_imbalance'].shift(i)
        
        # Rolling statistical features
        windows = [48, 96]  # 24 hours, 48 hours
        for window in windows:
            # Price statistics
            df[f'price_ma_{window}'] = df['system_price'].rolling(window=window).mean()
            df[f'price_std_{window}'] = df['system_price'].rolling(window=window).std()
            # Volume statistics
            df[f'volume_ma_{window}'] = df['trading_volume'].rolling(window=window).mean()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features(self, df):
        """Prepare model input features."""
        features = [
            'system_price',
            'hour', 'minute', 'time_code',
            'trading_volume', 'volume_imbalance', 'volume_ratio',
        ]
        
        # Add created features
        lag_features = [col for col in df.columns if 'lag_' in col]
        ma_features = [col for col in df.columns if 'ma_' in col or 'std_' in col]
        
        features.extend(lag_features)
        features.extend(ma_features)
        
        self.feature_columns = features
        X = df[features]
        y = df['system_price']
        
        return X, y
    
    def scale_data(self, X):
        """Scale data using MinMaxScaler."""
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def create_sequences(self, X_scaled, y, sequence_length):
        """Create sequences for LSTM input."""
        X_seq = []
        y_seq = []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            out, hidden = self.lstm(x)
            self.hidden_states = out  # Save hidden states for visualization
            out = out[:, -1, :]  # Get the last time step output
            out = self.fc(out)
            return out.squeeze()
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model."""
        # Initialize the model
        input_size = X_train.shape[2]
        self.model = self.LSTMModel(input_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        num_epochs = 100
        best_val_loss = np.inf
        patience = 10
        patience_counter = 0
        gradient_norms = []
        
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            epoch_grad_norm = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                
                # calculate gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norm.append(total_norm)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_grad_norm = np.mean(epoch_grad_norm)
            self.training_losses.append(avg_train_loss)
            gradient_norms.append(avg_grad_norm)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                self.validation_losses.append(val_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, Avg Grad Norm: {avg_grad_norm:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping")
                        break
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Avg Grad Norm: {avg_grad_norm:.4f}")
        
        # Load the best model
        if X_val is not None and y_val is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
        
        # Plot gradient norms
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(gradient_norms)+1), gradient_norms, marker='o')
        plt.title('Average Gradient Norm per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'gradient_norms.png'))
        plt.close()
    
    def predict(self, X, return_hidden_states=False):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            if return_hidden_states and hasattr(self.model, 'hidden_states'):
                hidden_states = self.model.hidden_states.cpu().numpy()
                return predictions.flatten(), hidden_states
        if return_hidden_states:
            return predictions.flatten(), None
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
                
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label='Training Loss')
        if self.validation_losses:
            plt.plot(self.validation_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'))
        plt.close()
    
    def visualize_hidden_states(self, hidden_states):
        """Visualize hidden states over time."""
        if hidden_states is None:
            print("No hidden states to visualize.")
            return
        # get the average hidden states over the batch
        avg_hidden_states = hidden_states.mean(axis=2)
        plt.figure(figsize=(15, 6))
        plt.imshow(avg_hidden_states.T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Hidden States Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Hidden Unit')
        plt.savefig(os.path.join(self.plots_dir, 'hidden_states.png'))
        plt.close()
    
    def add_attention_layer(self):
        """Add an attention mechanism to the LSTM model."""
        # Attention layer
        class LSTMAttentionModel(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.attention = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # calculate attention weights
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                # calculate context vector
                context = torch.sum(attn_weights * lstm_out, dim=1)
                out = self.fc(context)
                self.attention_weights = attn_weights
                return out.squeeze()
        
        # Initialize the model with attention
        input_size = len(self.feature_columns)
        self.model = LSTMAttentionModel(input_size).to(self.device)
        print("Attention layer added to the model.")
    
    def visualize_predictions(self, X_test, y_test, predictions):
        """Create visualizations for predictions and errors."""
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 1. Predictions vs Actual Values
        plt.figure(figsize=(15, 8))
        plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        
        # Add 95% confidence interval
        std_dev = np.std(y_test - predictions)
        plt.fill_between(range(len(predictions)),
                         predictions - 1.96*std_dev,
                         predictions + 1.96*std_dev,
                         alpha=0.2, color='red',
                         label='95% Confidence Interval')
        
        plt.title('Electricity Price Prediction vs Actual Values')
        plt.xlabel('Time Period')
        plt.ylabel('Price (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'prediction_with_ci.png'))
        plt.close()
        
        # 2. Prediction Error Analysis
        errors = y_test - predictions
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(errors, color='red', alpha=0.7)
        plt.title('Prediction Errors Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Error (JPY/kWh)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        sns.histplot(errors, bins=50, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Error (JPY/kWh)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'error_analysis.png'))
        plt.close()
        
        # 3. Predicted vs Actual Scatter Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', alpha=0.8)
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Price (JPY/kWh)')
        plt.ylabel('Predicted Price (JPY/kWh)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'scatter_plot.png'))
        plt.close()
        
        return errors
    
    def plot_predictions_over_time(self, y_test, predictions):
        """Plot actual vs predicted values over time."""
        plt.figure(figsize=(15, 8))
        plt.plot(y_test, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        plt.title('Actual vs Predicted Prices Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Price (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'predictions_over_time.png'))
        plt.close()
    
    def plot_residuals(self, errors):
        """Plot residuals to analyze prediction errors."""
        plt.figure(figsize=(15, 8))
        plt.plot(errors, label='Residuals', color='purple')
        plt.title('Residuals Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Residual (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'residuals_over_time.png'))
        plt.close()
        
        # Residuals Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50, kde=True, color='purple')
        plt.title('Residuals Distribution')
        plt.xlabel('Residual (JPY/kWh)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'residuals_distribution.png'))
        plt.close()
    
    def plot_residuals_acf_pacf(self, errors):
        """Plot ACF and PACF of residuals."""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        sample_size = len(errors)
        max_allowed_lags = int(sample_size * 0.5) - 1
        requested_lags = 50
        lags = min(requested_lags, max_allowed_lags)

        if lags < 1:
            lags = 1  

        print(f"Plotting ACF and PACF with lags={lags} based on sample size={sample_size}")

        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plot_acf(errors, lags=lags, ax=plt.gca())
        plt.title('Residuals Autocorrelation')
        
        plt.subplot(1, 2, 2)
        plot_pacf(errors, lags=lags, ax=plt.gca(), method='ywm')
        plt.title('Residuals Partial Autocorrelation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residuals_acf_pacf.png'))
        plt.close()
    
    def visualize_attention_weights(self, X_test):
        """Visualize attention weights."""
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _ = self.model(X_tensor)
            if hasattr(self.model, 'attention_weights'):
                attn_weights = self.model.attention_weights.cpu().numpy()
            else:
                print("No attention weights to visualize.")
                return
        
        avg_attn_weights = attn_weights.mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(avg_attn_weights)), avg_attn_weights.squeeze())
        plt.title('Average Attention Weights Over Time Steps')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(self.plots_dir, 'attention_weights.png'))
        plt.close()
    
    def perform_residuals_diagnostics(self, errors):
        """Perform statistical tests on residuals."""
        shapiro_test = shapiro(errors)
        normal_test_stat, normal_test_p = normaltest(errors)
        
        adf_result = adfuller(errors)
        
        report_content = f"""
    # Residuals Diagnostics Report

    ## Normality Tests
    - Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}
    - D'Agostino's K-squared Test Statistic: {normal_test_stat:.4f}, p-value: {normal_test_p:.4f}

    ## Stationarity Test
    - Augmented Dickey-Fuller Test Statistic: {adf_result[0]:.4f}
    - p-value: {adf_result[1]:.4f}
    - Critical Values:
        - 1%: {adf_result[4]['1%']:.4f}
        - 5%: {adf_result[4]['5%']:.4f}
        - 10%: {adf_result[4]['10%']:.4f}
    """
        # Save report
        self.generate_report(report_content, report_name='residuals_diagnostics.md')
        print("Residuals diagnostics report generated.")
    
    def generate_report(self, report_content, report_name='report.md'):
        """Generate markdown report."""
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, report_name)
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"Report saved to {report_path}")
    
    def generate_prediction_report(self, X_test, y_test, predictions, metrics):
        """Generate detailed prediction report."""
        report = {
            'Overall Metrics': {
                'Mean Squared Error': metrics['mse'],
                'Root Mean Squared Error': metrics['rmse'],
                'Mean Absolute Error': metrics['mae'],
                'R² Score': metrics['r2'],
                'Mean Prediction': np.mean(predictions),
                'Std Prediction': np.std(predictions)
            },
            'Price Statistics': {
                'Actual Mean': np.mean(y_test),
                'Actual Std': np.std(y_test),
                'Actual Min': np.min(y_test),
                'Actual Max': np.max(y_test),
                'Predicted Min': np.min(predictions),
                'Predicted Max': np.max(predictions)
            },
            'Error Analysis': {
                'Mean Error': np.mean(y_test - predictions),
                'Error Std': np.std(y_test - predictions),
                'Max Underestimation': np.min(y_test - predictions),
                'Max Overestimation': np.max(y_test - predictions),
                'Error Range': np.ptp(y_test - predictions)
            }
        }
        
        # Convert to DataFrame and save
        report_df = pd.DataFrame({k: v for d in report.values() for k, v in d.items()}, 
                                 index=[0])
        report_df = report_df.round(4)
        report_df.to_csv(os.path.join(self.reports_dir, 'prediction_report.csv'), index=False)
        
        # Generate markdown report
        report_content = f"""
    # Detailed Prediction Report

    ## Overall Metrics
    - Mean Squared Error: {metrics['mse']:.4f}
    - Root Mean Squared Error: {metrics['rmse']:.4f}
    - Mean Absolute Error: {metrics['mae']:.4f}
    - R² Score: {metrics['r2']:.4f}

    ## Price Statistics
    - Actual Mean: {report['Price Statistics']['Actual Mean']:.4f}
    - Actual Std: {report['Price Statistics']['Actual Std']:.4f}
    - Actual Min: {report['Price Statistics']['Actual Min']:.4f}
    - Actual Max: {report['Price Statistics']['Actual Max']:.4f}
    - Predicted Min: {report['Price Statistics']['Predicted Min']:.4f}
    - Predicted Max: {report['Price Statistics']['Predicted Max']:.4f}

    ## Error Analysis
    - Mean Error: {report['Error Analysis']['Mean Error']:.4f}
    - Error Std: {report['Error Analysis']['Error Std']:.4f}
    - Max Underestimation: {report['Error Analysis']['Max Underestimation']:.4f}
    - Max Overestimation: {report['Error Analysis']['Max Overestimation']:.4f}
    - Error Range: {report['Error Analysis']['Error Range']:.4f}
    """
        # Save report
        self.generate_report(report_content, report_name='prediction_report.md')
        
        # Print report summary
        print("\nDetailed Prediction Report:")
        for section, metrics_dict in report.items():
            print(f"\n{section}:")
            for metric, value in metrics_dict.items():
                print(f"{metric}: {value:.4f}")
                    
        return report
    
    # plot attention weights
    def plot_attention_weights_sample(self, X_sample, y_sample, index=0):
        """Plot attention weights for a specific sample."""
        if not hasattr(self.model, 'attention_weights'):
            print("No attention weights available.")
            return
        
        attn_weights = self.model.attention_weights[index].cpu().numpy().squeeze()
        time_steps = range(len(attn_weights))
        
        plt.figure(figsize=(12, 6))
        plt.bar(time_steps, attn_weights, color='skyblue')
        plt.title(f'Attention Weights for Sample {index}')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'attention_weights_sample_{index}.png'))
        plt.close()
    
    def plot_hidden_state_dynamics(self, hidden_states, units=[0, 1, 2]):
        """Plot dynamics of selected hidden units over time."""
        if hidden_states is None:
            print("No hidden states to visualize.")
            return
        
        # hidden_states: (batch_size, seq_length, hidden_size)
        # Compute average over batch for each time step and selected units
        avg_hidden_states = hidden_states.mean(axis=0)  # (seq_length, hidden_size)
        
        plt.figure(figsize=(15, 6))
        for unit in units:
            plt.plot(avg_hidden_states[:, unit], label=f'Hidden Unit {unit}')
        
        plt.title('Dynamics of Selected Hidden Units')
        plt.xlabel('Time Step')
        plt.ylabel('Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'hidden_state_dynamics.png'))
        plt.close()
    
    def plot_sequence_prediction(self, X_sequence, y_true, y_pred, sequence_length=48, index=0):
        """Plot input sequence, actual value, and predicted value for a specific sample."""
        plt.figure(figsize=(15, 8))
        
        # Plot input sequence
        system_price_index = self.feature_columns.index('system_price')
        plt.plot(range(sequence_length), X_sequence[index, :, system_price_index], label='Input Sequence (System Price)', color='gray')
        
        # Plot actual and predicted values
        plt.plot(sequence_length, y_true[index], 'go', label='Actual Price')
        plt.plot(sequence_length, y_pred[index], 'ro', label='Predicted Price')
        
        plt.title(f'Sequence Prediction for Sample {index}')
        plt.xlabel('Time Step')
        plt.ylabel('Price (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'sequence_prediction_sample_{index}.png'))
        plt.close()
    
    def plot_hidden_state_correlation(self, hidden_states, X_test, feature_indices=[0, 1, 2]):
        """Plot correlation between hidden states and selected input features."""
        if hidden_states is None:
            print("No hidden states to visualize.")
            return

        # hidden_states: (batch_size, seq_length, hidden_size)
        # X_test: (batch_size, seq_length, num_features)
        # Flatten batch_size and seq_length dimensions
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[2])  # (batch_size * seq_length, hidden_size)
        X_test_flat = X_test.reshape(-1, X_test.shape[2])  # (batch_size * seq_length, num_features)

        correlations = {}
        for feature_idx in feature_indices:
            feature_name = self.feature_columns[feature_idx]
            correlations[feature_name] = [
                np.corrcoef(X_test_flat[:, feature_idx], hidden_states_flat[:, unit])[0, 1] 
                for unit in range(hidden_states_flat.shape[1])
            ]
        
        # Plotting
        plt.figure(figsize=(12, 6))
        for feature, corr in correlations.items():
            plt.plot(range(len(corr)), corr, label=feature)
        
        plt.title('Correlation between Hidden States and Input Features')
        plt.xlabel('Hidden Unit')
        plt.ylabel('Correlation Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'hidden_state_correlation.png'))
        plt.close()
    
    def plot_hidden_states_pca(self, hidden_states, n_components=2):
        """Perform PCA on hidden states and plot the first two principal components."""
        if hidden_states is None:
            print("No hidden states to visualize.")
            return
        
        # hidden_states: (batch_size, seq_length, hidden_size)
        # Flatten batch_size and seq_length dimensions
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[2])  # (batch_size * seq_length, hidden_size)
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(hidden_states_flat)
        
        plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA of Hidden States')
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.5)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D PCA of Hidden States')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'hidden_states_pca.png'))
        plt.close()
    
    def plot_shap_feature_importance(self, X_sample, y_sample):
        """Plot SHAP feature importance for LSTM model."""
        try:
            # Select a subset of data for SHAP
            background = X_sample[:100]
            explainer = shap.GradientExplainer(self.model, background)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot summary
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_columns, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'shap_summary.png'))
            plt.close()
            print("SHAP summary plot saved.")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    def perform_shap_analysis(self, X_sample, y_sample):
        """Perform SHAP analysis on the LSTM model."""
        self.plot_shap_feature_importance(X_sample, y_sample)
    
    def plot_residuals_decomposition(self, errors, freq=48):
        """Decompose residuals into trend, seasonal, and residual components."""
        decomposition = seasonal_decompose(errors, model='additive', period=freq)
        
        plt.figure(figsize=(15, 8))
        decomposition.plot()
        plt.suptitle('Residuals Decomposition', fontsize=16)
        plt.savefig(os.path.join(self.plots_dir, 'residuals_decomposition.png'))
        plt.close()
    
    def plot_residuals_dynamics(self, errors):
        """Decompose residuals and plot the components."""
        decomposition = seasonal_decompose(errors, model='additive', period=48)
        
        plt.figure(figsize=(15, 8))
        plt.subplot(411)
        plt.plot(errors, label='Residuals')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonal')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residual')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residuals_decomposition_detailed.png'))
        plt.close()
    
    def plot_hidden_states_heatmap(self, hidden_states):
        """Plot a heatmap of hidden states."""
        if hidden_states is None:
            print("No hidden states to visualize.")
            return
        
        # hidden_states: (batch_size, seq_length, hidden_size)
        # Compute average over batch and sequence
        avg_hidden_states = hidden_states.mean(axis=0)  # (seq_length, hidden_size)
        plt.figure(figsize=(15, 6))
        sns.heatmap(avg_hidden_states.T, cmap='viridis')
        plt.title('Heatmap of Average Hidden States')
        plt.xlabel('Time Step')
        plt.ylabel('Hidden Unit')
        plt.savefig(os.path.join(self.plots_dir, 'hidden_states_heatmap.png'))
        plt.close()
    
    def perform_sensitivity_analysis(self, df, variable, change_percentages):
        """Perform sensitivity analysis on a specific variable."""
        original_values = df[variable].copy()
        results = {}
        
        for pct_change in change_percentages:
            df[variable] = original_values * (1 + pct_change / 100)
            # Recreate features and scale
            df_scaled = self.scaler.transform(df[self.feature_columns])
            # Create sequences
            X_seq, _ = self.create_sequences(df_scaled, df['system_price'], self.sequence_length)
            # Predict
            predictions = self.predict(X_seq)
            average_price = predictions.mean()
            results[pct_change] = average_price
        
        # Restore original values
        df[variable] = original_values
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), marker='o')
        plt.title(f'Sensitivity Analysis on {variable}')
        plt.xlabel('Percentage Change (%)')
        plt.ylabel('Average Predicted Price (JPY/kWh)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'sensitivity_{variable}.png'))
        plt.close()
        
        # Return results
        return results


def main():
    try:
        # Initialize the forecaster
        forecaster = JPEXPriceForecastLSTM()
        
        print("Loading data...")
        df = forecaster.load_data('spot_summary_2024.csv')
        if df is None:
            return
        
        print("Creating features...")
        df = forecaster.create_features(df)
        
        print("Preparing features...")
        X, y = forecaster.prepare_features(df)
        
        print("Scaling data...")
        X_scaled = forecaster.scale_data(X)
        
        print("Creating sequences...")
        sequence_length = forecaster.sequence_length
        X_seq, y_seq = forecaster.create_sequences(X_scaled, y, sequence_length)
        
        print("Splitting data...")
        # Use the last two days of data as the test set
        test_size = 48 * 2  # 2 days, 48 time periods each day
        X_test = X_seq[-test_size:]
        y_test = y_seq[-test_size:]
        X_train_val = X_seq[:-test_size]
        y_train_val = y_seq[:-test_size]
        
        # Split the remaining data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        print("Adding attention layer to the model...")
        forecaster.add_attention_layer()
        
        print("Training model...")
        forecaster.train_model(X_train, y_train, X_val, y_val)
        
        print("Plotting training curves...")
        forecaster.plot_training_curves()
        
        print("Making predictions and extracting hidden states...")
        predictions, hidden_states = forecaster.predict(X_test, return_hidden_states=True)
        
        # Evaluate the model
        print("\nEvaluating model performance...")
        metrics = forecaster.evaluate(X_test, y_test)
        print(f"Test set metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        # Visualizations
        print("\nGenerating visualizations...")
        errors = forecaster.visualize_predictions(X_test, y_test, predictions)
        
        print("Plotting predictions over time...")
        forecaster.plot_predictions_over_time(y_test, predictions)
        
        print("Plotting residuals...")
        forecaster.plot_residuals(errors)
        
        print("Plotting residuals ACF and PACF...")
        forecaster.plot_residuals_acf_pacf(errors)
        
        print("Performing residuals diagnostics...")
        forecaster.perform_residuals_diagnostics(errors)
        
        print("Visualizing hidden states...")
        forecaster.visualize_hidden_states(hidden_states)
        
        print("Visualizing attention weights...")
        forecaster.visualize_attention_weights(X_test)
        
        # 新增可视化方法调用开始
        print("Plotting attention weights for sample 0...")
        forecaster.plot_attention_weights_sample(X_test, y_test, index=0)
        
        print("Plotting hidden state dynamics...")
        forecaster.plot_hidden_state_dynamics(hidden_states, units=[0, 1, 2])
        
        print("Plotting sequence prediction for sample 0...")
        forecaster.plot_sequence_prediction(X_test, y_test, predictions, index=0)
        
        print("Plotting hidden state correlation...")
        # Select specific features for correlation analysis
        feature_indices = [
            forecaster.feature_columns.index('system_price'), 
            forecaster.feature_columns.index('trading_volume'), 
            forecaster.feature_columns.index('volume_imbalance')
        ]
        forecaster.plot_hidden_state_correlation(hidden_states, X_test, feature_indices=feature_indices)
        
        print("Plotting hidden states PCA...")
        forecaster.plot_hidden_states_pca(hidden_states, n_components=2)
        
        print("Performing SHAP feature importance analysis...")
        # Select a sample of data for SHAP analysis
        shap_sample_size = min(100, X_test.shape[0])
        forecaster.perform_shap_analysis(X_test[:shap_sample_size], y_test[:shap_sample_size])
        
        print("Plotting residuals decomposition...")
        forecaster.plot_residuals_decomposition(errors)
        
        print("Plotting residuals dynamics...")
        forecaster.plot_residuals_dynamics(errors)
        
        print("Plotting hidden states heatmap...")
        forecaster.plot_hidden_states_heatmap(hidden_states)

        # Generate detailed report
        print("\nGenerating detailed prediction report...")
        report = forecaster.generate_prediction_report(X_test, y_test, predictions, metrics)
        
        # Example of performing sensitivity analysis
        print("\nPerforming sensitivity analysis on 'trading_volume'...")
        change_percentages = [-10, -5, 0, 5, 10]
        sensitivity_results = forecaster.perform_sensitivity_analysis(df, 'trading_volume', change_percentages)
        print("Sensitivity Analysis Results:")
        for pct, avg_price in sensitivity_results.items():
            print(f"Change: {pct}%, Average Predicted Price: {avg_price:.4f} JPY/kWh")
        
        print("\nAll results have been saved to the 'outputs_LSTM' directory.")
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'errors': errors,
            'report': report
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
