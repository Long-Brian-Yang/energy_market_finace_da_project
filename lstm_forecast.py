import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# For statistical tests
from statsmodels.tsa.stattools import coint, adfuller

class JPEXPriceForecastLSTM:
    def __init__(self, output_dir='outputs_LSTM'):
        """Initialize the forecaster."""
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.output_dir = output_dir
        self.create_output_directories()
        self.sequence_length = 48  # 使用过去 24 小时的数据（48 个时间步长）进行预测
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
        
        # 打印原始 DataFrame 的列名以进行调试
        print("Columns in the original DataFrame:", df.columns.tolist())
        
        # 更新列名映射
        columns = {
            '日付': 'date',
            '時刻コード': 'time_code',
            '売り入札量(kWh)': 'sell_volume',
            '買い入札量(kWh)': 'buy_volume',
            '約定総量(kWh)': 'trading_volume',
            'システムプライス(円/kWh)': 'system_price'
        }
        df = df.rename(columns=columns)
        
        # 检查是否存在 'date' 列
        if 'date' not in df.columns:
            # 如果没有，可以手动添加或抛出错误
            df['date'] = '2024-01-01'  # 替换为实际日期
            df['date'] = pd.to_datetime(df['date'])
            print("Added 'date' column manually.")
        else:
            df['date'] = pd.to_datetime(df['date'])
            print("Converted 'date' column to datetime.")
        
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
        
        # Historical statistics for each time block
        df['price_time_block_mean'] = df.groupby('time_code')['system_price'].transform('mean')
        df['price_time_block_std'] = df.groupby('time_code')['system_price'].transform('std')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features(self, df):
        """Prepare model input features."""
        features = [
            'system_price',  # LSTM 输入需要包含目标变量
            'hour', 'minute', 'time_code',
            'trading_volume', 'volume_imbalance', 'volume_ratio',
            'price_time_block_mean', 'price_time_block_std'
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
            self.hidden_states = out  # 保存所有时间步的隐藏状态
            out = out[:, -1, :]  # Get the last time step output
            out = self.fc(out)
            return out.squeeze()
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model."""
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
        
        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                # 梯度剪裁，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.training_losses.append(avg_train_loss)
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                self.validation_losses.append(val_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")
        
        # Load the best model
        if X_val is not None and y_val is not None:
            self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
    
    def predict(self, X, return_hidden_states=False):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            if return_hidden_states:
                hidden_states = self.model.hidden_states.cpu().numpy()
                return predictions.flatten(), hidden_states
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
        # 取平均值或者其他方式处理隐藏状态
        avg_hidden_states = hidden_states.mean(axis=2)
        plt.figure(figsize=(15, 6))
        plt.imshow(avg_hidden_states.T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Hidden States Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Hidden Unit')
        plt.savefig(os.path.join(self.plots_dir, 'hidden_states.png'))
        plt.close()
    
    def analyze_time_step_importance(self, X_test, y_test):
        """Analyze the importance of different time steps."""
        # 遍历时间步长，评估每个时间步长的重要性
        sequence_length = X_test.shape[1]
        importances = []
        base_predictions = self.predict(X_test)
        for t in range(sequence_length):
            X_modified = X_test.copy()
            X_modified[:, t, :] = 0  # 将第 t 个时间步的输入置零
            modified_predictions = self.predict(X_modified)
            importance = mean_squared_error(base_predictions, modified_predictions)
            importances.append(importance)
        
        # 可视化时间步长的重要性
        plt.figure(figsize=(12, 6))
        plt.bar(range(sequence_length), importances)
        plt.title('Time Step Importance')
        plt.xlabel('Time Step')
        plt.ylabel('MSE Increase')
        plt.savefig(os.path.join(self.plots_dir, 'time_step_importance.png'))
        plt.close()
    
    def detect_gradient_flow(self):
        """Detect gradient flow to check for vanishing or exploding gradients."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Total Gradient Norm: {total_norm}")
    
    def add_attention_layer(self):
        """Add an attention mechanism to the LSTM model."""
        # 注意力机制需要修改模型结构
        class LSTMAttentionModel(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.attention = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # 计算注意力权重
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                # 加权求和
                context = torch.sum(attn_weights * lstm_out, dim=1)
                out = self.fc(context)
                self.attention_weights = attn_weights
                return out.squeeze()
        
        # 用新的模型替换旧模型
        input_size = self.feature_columns.__len__()
        self.model = LSTMAttentionModel(input_size).to(self.device)
        print("Attention layer added to the model.")
    
    def visualize_attention_weights(self, X_test):
        """Visualize attention weights."""
        # 获取注意力权重
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _ = self.model(X_tensor)
            attn_weights = self.model.attention_weights.cpu().numpy()
        
        avg_attn_weights = attn_weights.mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(avg_attn_weights)), avg_attn_weights.squeeze())
        plt.title('Average Attention Weights Over Time Steps')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(self.plots_dir, 'attention_weights.png'))
        plt.close()
    
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
        for section, metrics in report.items():
            print(f"\n{section}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        return report
    
    # 添加经济学分析方法，从您的 XGBoost 代码中复制
    def plot_supply_demand_curves(self, df, time_period=None):
        """Plot supply and demand curves for a specific time period."""
        # If time_period is None, use the last available period
        if time_period is None:
            time_period = df['datetime'].iloc[-1]
        else:
            time_period = pd.to_datetime(time_period)
        
        # Filter data for the specified time period
        df_period = df[df['datetime'] == time_period]
        
        if df_period.empty:
            print(f"No data available for {time_period}")
            return
        
        # Assuming 'system_price', 'sell_volume', and 'buy_volume' are available
        price_points = df_period['system_price'].unique()
        sell_volumes = df_period.groupby('system_price')['sell_volume'].sum().sort_index()
        buy_volumes = df_period.groupby('system_price')['buy_volume'].sum().sort_index(ascending=False)
        
        cumulative_supply = sell_volumes.cumsum()
        cumulative_demand = buy_volumes.cumsum()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_supply, sell_volumes.index, label='Supply Curve', drawstyle='steps-post')
        plt.plot(cumulative_demand, buy_volumes.index, label='Demand Curve', drawstyle='steps-post')
        
        # Equilibrium point (approximate)
        equilibrium_price = df_period['system_price'].iloc[0]
        equilibrium_quantity = cumulative_supply.iloc[0]
        
        plt.axhline(equilibrium_price, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(equilibrium_quantity, color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f'Supply and Demand Curves on {time_period}')
        plt.xlabel('Cumulative Quantity (kWh)')
        plt.ylabel('Price (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'supply_demand_{time_period}.png'))
        plt.close()
    
    def calculate_price_elasticity(self, df):
        """Calculate and plot price elasticity of demand."""
        # Calculate percentage changes
        df['pct_change_price'] = df['system_price'].pct_change()
        df['pct_change_demand'] = df['buy_volume'].pct_change()
        
        # Avoid division by zero
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pct_change_price', 'pct_change_demand'])
        
        # Calculate elasticity
        df['elasticity'] = df['pct_change_demand'] / df['pct_change_price']
        
        # Plot elasticity over time
        plt.figure(figsize=(15, 8))
        plt.plot(df['datetime'], df['elasticity'], label='Price Elasticity of Demand')
        plt.title('Price Elasticity of Demand Over Time')
        plt.xlabel('Time')
        plt.ylabel('Elasticity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'price_elasticity.png'))
        plt.close()
        
        # Return elasticity DataFrame
        return df[['datetime', 'elasticity']]
    
    def analyze_price_volatility(self, df):
        """Analyze and plot price volatility."""
        # Calculate rolling standard deviation
        df['price_volatility'] = df['system_price'].rolling(window=48).std()
        
        # Plot volatility over time
        plt.figure(figsize=(15, 8))
        plt.plot(df['datetime'], df['price_volatility'], label='Price Volatility')
        plt.title('Electricity Price Volatility Over Time')
        plt.xlabel('Time')
        plt.ylabel('Volatility (Standard Deviation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'price_volatility.png'))
        plt.close()
        
        # Return volatility DataFrame
        return df[['datetime', 'price_volatility']]
    
    def analyze_seasonal_patterns(self, df):
        """Analyze and plot seasonal patterns."""
        # Extract date components
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'] >= 5
        
        # Monthly average prices
        monthly_prices = df.groupby('month')['system_price'].mean()
        
        # Plot monthly average prices
        plt.figure(figsize=(10, 6))
        monthly_prices.plot(kind='bar')
        plt.title('Average Electricity Price by Month')
        plt.xlabel('Month')
        plt.ylabel('Price (JPY/kWh)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'monthly_average_prices.png'))
        plt.close()
        
        # Weekday vs Weekend prices
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='is_weekend', y='system_price', data=df)
        plt.title('Electricity Price: Weekday vs Weekend')
        plt.xlabel('Is Weekend')
        plt.ylabel('Price (JPY/kWh)')
        plt.savefig(os.path.join(self.plots_dir, 'weekday_weekend_prices.png'))
        plt.close()
        
        # Return DataFrame with seasonal features
        return df
    
    def analyze_profit_margins(self, df, marginal_cost):
        """Analyze profit margins given a marginal cost."""
        df['profit_margin'] = df['system_price'] - marginal_cost
        
        # Plot profit margins over time
        plt.figure(figsize=(15, 8))
        plt.plot(df['datetime'], df['profit_margin'], label='Profit Margin')
        plt.title('Profit Margins Over Time')
        plt.xlabel('Time')
        plt.ylabel('Profit Margin (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'profit_margins.png'))
        plt.close()
        
        # Return DataFrame with profit margins
        return df
    
    def plot_load_duration_curve(self, df):
        """Plot the load duration curve."""
        # Sort demand data
        sorted_demand = df['buy_volume'].sort_values(ascending=False).reset_index(drop=True)
        time_percentile = np.linspace(0, 100, len(sorted_demand))
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time_percentile, sorted_demand)
        plt.title('Load Duration Curve')
        plt.xlabel('Time Percentile (%)')
        plt.ylabel('Demand (kWh)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'load_duration_curve.png'))
        plt.close()
    
    def perform_sensitivity_analysis(self, df, variable, change_percentages):
        """Perform sensitivity analysis on a specific variable."""
        original_values = df[variable].copy()
        results = {}
        
        for pct_change in change_percentages:
            df[variable] = original_values * (1 + pct_change / 100)
            # 需要重新处理特征并进行预测
            X_modified, _ = self.prepare_features(df)
            X_scaled_modified = self.scaler.transform(X_modified)
            X_seq_modified, _ = self.create_sequences(X_scaled_modified, df['system_price'], self.sequence_length)
            predictions = self.predict(X_seq_modified)
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
    
    def forecast_demand(self, df):
        """Forecast future electricity demand."""
        # Use historical demand data
        demand_series = df['buy_volume']
        
        # Simple Moving Average Forecast as an example
        df['demand_forecast'] = demand_series.rolling(window=48).mean().shift(1)
        
        # Plot forecast vs actual
        plt.figure(figsize=(15, 8))
        plt.plot(df['datetime'], df['buy_volume'], label='Actual Demand')
        plt.plot(df['datetime'], df['demand_forecast'], label='Forecasted Demand', alpha=0.7)
        plt.title('Electricity Demand Forecast')
        plt.xlabel('Time')
        plt.ylabel('Demand (kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'demand_forecast.png'))
        plt.close()
        
        # Return DataFrame with forecast
        return df

def main():
    try:
        # Initialize the forecaster
        forecaster = JPEXPriceForecastLSTM()
        
        print("Loading data...")
        df = forecaster.load_data('spot_summary_2024.csv')
        
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
        hourly_metrics = forecaster.visualize_predictions(X_test, y_test, predictions)
        
        print("Visualizing hidden states...")
        forecaster.visualize_hidden_states(hidden_states)
        
        print("Visualizing attention weights...")
        forecaster.visualize_attention_weights(X_test)
        
        print("Analyzing time step importance...")
        forecaster.analyze_time_step_importance(X_test, y_test)
        
        # Generate detailed report
        print("\nGenerating detailed prediction report...")
        report = forecaster.generate_prediction_report(X_test, y_test, predictions, metrics)
        
        # Additional Economic Analyses
        print("\nPerforming additional economic analyses...")
        
        # 1. Supply and Demand Curve
        forecaster.plot_supply_demand_curves(df)
        
        # 2. Price Elasticity of Demand
        elasticity_df = forecaster.calculate_price_elasticity(df)
        
        # 3. Price Volatility Analysis
        volatility_df = forecaster.analyze_price_volatility(df)
        
        # 4. Seasonal Patterns
        df = forecaster.analyze_seasonal_patterns(df)
        
        # 5. Profit Margins
        df = forecaster.analyze_profit_margins(df, marginal_cost=10)
        
        # 6. Load Duration Curve
        forecaster.plot_load_duration_curve(df)
        
        # 7. Sensitivity Analysis
        change_percentages = [-10, -5, 0, 5, 10]
        sensitivity_results = forecaster.perform_sensitivity_analysis(df, 'trading_volume', change_percentages)
        
        # 8. Demand Forecasting
        df = forecaster.forecast_demand(df)
        
        print("\nAll results have been saved to the 'outputs_LSTM' directory.")
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'hourly_metrics': hourly_metrics,
            'report': report,
            'elasticity_df': elasticity_df,
            'volatility_df': volatility_df,
            'sensitivity_results': sensitivity_results
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
