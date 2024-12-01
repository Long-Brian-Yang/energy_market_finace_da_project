import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# For statistical tests
from statsmodels.tsa.stattools import coint, adfuller

class JPEXPriceForecastXGB:
    def __init__(self, output_dir='outputs_XGB'):
        """Initialize the forecaster."""
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.output_dir = output_dir
        self.create_output_directories()
        
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
        
        # Rename columns
        columns = {
            '日付': 'date',  # 确保这个列名与您的数据匹配
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
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        self.model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=100
        )
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self.model.predict(X)
    
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
    
    def plot_feature_importance(self):
        """Plot feature importance."""
        importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance.head(20), x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'))
        plt.close()
        
        return importance
    
    def visualize_predictions(self, X_test, y_test, predictions):
        """Create visualizations for predictions and errors."""
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 1. Predictions vs Actual Values
        plt.figure(figsize=(15, 8))
        plt.plot(y_test.values, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
        
        # Add 95% confidence interval
        std_dev = np.std(y_test.values - predictions)
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
        errors = y_test.values - predictions
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
        
        # 3. Hourly Analysis
        df_analysis = pd.DataFrame({
            'Hour': X_test['hour'],
            'Actual': y_test.values,
            'Predicted': predictions,
            'Error': errors
        })
        
        # Aggregate error metrics by hour
        hourly_metrics = df_analysis.groupby('Hour').agg({
            'Error': ['mean', 'std'],
            'Actual': 'mean',
            'Predicted': 'mean'
        }).round(4)
        
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df_analysis, x='Hour', y='Error')
        plt.title('Error Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Error (JPY/kWh)')
        
        plt.subplot(2, 1, 2)
        hourly_metrics[('Error', 'mean')].plot(kind='bar', 
                                               yerr=hourly_metrics[('Error', 'std')],
                                               capsize=5, alpha=0.7)
        plt.title('Mean Error and Standard Deviation by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Error (JPY/kWh)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'hourly_analysis.png'))
        plt.close()
        
        # 4. Predicted vs Actual Scatter Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test.values, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', alpha=0.8)
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Price (JPY/kWh)')
        plt.ylabel('Predicted Price (JPY/kWh)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'scatter_plot.png'))
        plt.close()
        
        return hourly_metrics
    
    def generate_report(self, report_content, report_name='report.md'):
        """Generate markdown report."""
        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, report_name)
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"Report saved to {report_path}")
    
    def predict_specific_date(self, df, target_date_str='2024-12-07'):
        """Predict electricity prices for a specific date."""
        # Create features for 48 time periods
        future_data = []
        base_date = pd.to_datetime(target_date_str)
        
        for time_code in range(1, 49):
            hour = (time_code - 1) // 2
            minute = ((time_code - 1) % 2) * 30
            time_dict = {
                'time_code': time_code,
                'hour': hour,
                'minute': minute,
                # You can add more time-related features here
            }
            
            # Use recent historical data to fill other features
            recent_data = df.iloc[-1].copy()
            for key in ['trading_volume', 'volume_imbalance', 'volume_ratio']:
                if key in df.columns:
                    time_dict[key] = recent_data[key]
            
            # Add time block statistical features
            time_block_stats = df[df['time_code'] == time_code]['system_price']
            time_dict['price_time_block_mean'] = time_block_stats.mean()
            time_dict['price_time_block_std'] = time_block_stats.std()
            
            # For lag and moving average features, use recent values or set to mean
            for feature in self.feature_columns:
                if feature not in time_dict:
                    if feature.startswith('price_lag_') or feature.startswith('volume_lag_') or feature.startswith('imbalance_lag_'):
                        time_dict[feature] = df[feature].iloc[-1]
                    elif feature.startswith('price_ma_') or feature.startswith('price_std_') or feature.startswith('volume_ma_'):
                        time_dict[feature] = df[feature].iloc[-1]
                    else:
                        time_dict[feature] = df[feature].mean()
            
            future_data.append(time_dict)
        
        future_df = pd.DataFrame(future_data)
        
        # Generate predictions
        predictions = self.predict(future_df[self.feature_columns])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'datetime': [base_date + timedelta(hours=hour, minutes=minute) for hour, minute in zip(future_df['hour'], future_df['minute'])],
            'time_code': future_df['time_code'],
            'hour': future_df['hour'],
            'predicted_price': predictions,
            'confidence_lower': predictions * 0.95,
            'confidence_upper': predictions * 1.05
        })
        
        return results
    
    def analyze_future_predictions(self, predictions_df):
        """Analyze future prediction results."""
        # Ensure plots and reports directories exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 1. Price Trend Plot
        plt.figure(figsize=(15, 8))
        plt.plot(predictions_df['datetime'], 
                 predictions_df['predicted_price'], 
                 label='Predicted Price', 
                 color='blue', marker='o')
        plt.fill_between(predictions_df['datetime'],
                         predictions_df['confidence_lower'],
                         predictions_df['confidence_upper'],
                         alpha=0.2, color='blue',
                         label='95% Confidence Interval')
        plt.title('Predicted Electricity Prices for December 7, 2024')
        plt.xlabel('Time')
        plt.ylabel('Price (JPY/kWh)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'future_price_forecast.png'))
        plt.close()
        
        # 2. Intraday Price Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='predicted_price', data=predictions_df)
        plt.title('Predicted Price Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Price (JPY/kWh)')
        plt.savefig(os.path.join(self.plots_dir, 'future_hourly_distribution.png'))
        plt.close()
        
        # 3. Statistical Analysis
        stats = {
            'Overall Statistics': {
                'Mean Price': predictions_df['predicted_price'].mean(),
                'Std Dev': predictions_df['predicted_price'].std(),
                'Min Price': predictions_df['predicted_price'].min(),
                'Max Price': predictions_df['predicted_price'].max(),
                'Price Range': predictions_df['predicted_price'].max() - 
                               predictions_df['predicted_price'].min()
            },
            'Peak Hours (8:00-20:00)': {
                'Mean Price': predictions_df[
                    predictions_df['hour'].between(8, 19)]['predicted_price'].mean(),
                'Max Price': predictions_df[
                    predictions_df['hour'].between(8, 19)]['predicted_price'].max(),
                'Peak Hour': predictions_df.loc[
                    predictions_df['predicted_price'].idxmax(), 'hour']
            },
            'Off-Peak Hours': {
                'Mean Price': predictions_df[
                    ~predictions_df['hour'].between(8, 19)]['predicted_price'].mean(),
                'Min Price': predictions_df[
                    ~predictions_df['hour'].between(8, 19)]['predicted_price'].min(),
                'Lowest Hour': predictions_df.loc[
                    predictions_df['predicted_price'].idxmin(), 'hour']
            }
        }
        
        # Save detailed prediction results
        predictions_df.to_csv(os.path.join(self.reports_dir, 'future_detailed_predictions.csv'), index=False)
        
        # Create hourly statistics
        hourly_stats = predictions_df.groupby('hour')['predicted_price'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(2)
        hourly_stats.to_csv(os.path.join(self.reports_dir, 'future_hourly_statistics.csv'))
        
        # Generate markdown report
        report = f"""
# Electricity Price Forecast Report for December 7, 2024

## Overall Statistics
- Mean Price: {stats['Overall Statistics']['Mean Price']:.2f} JPY/kWh
- Standard Deviation: {stats['Overall Statistics']['Std Dev']:.2f} JPY/kWh
- Price Range: {stats['Overall Statistics']['Price Range']:.2f} JPY/kWh
- Minimum Price: {stats['Overall Statistics']['Min Price']:.2f} JPY/kWh
- Maximum Price: {stats['Overall Statistics']['Max Price']:.2f} JPY/kWh

## Peak Hours Analysis (8:00-20:00)
- Average Peak Price: {stats['Peak Hours (8:00-20:00)']['Mean Price']:.2f} JPY/kWh
- Maximum Peak Price: {stats['Peak Hours (8:00-20:00)']['Max Price']:.2f} JPY/kWh
- Hour with Highest Price: {int(stats['Peak Hours (8:00-20:00)']['Peak Hour']):02d}:00

## Off-Peak Hours Analysis
- Average Off-Peak Price: {stats['Off-Peak Hours']['Mean Price']:.2f} JPY/kWh
- Minimum Off-Peak Price: {stats['Off-Peak Hours']['Min Price']:.2f} JPY/kWh
- Hour with Lowest Price: {int(stats['Off-Peak Hours']['Lowest Hour']):02d}:00

## Key Findings
1. The highest prices are expected during {int(stats['Peak Hours (8:00-20:00)']['Peak Hour']):02d}:00.
2. The lowest prices are expected during {int(stats['Off-Peak Hours']['Lowest Hour']):02d}:00.
3. The average price difference between peak and off-peak hours is {stats['Peak Hours (8:00-20:00)']['Mean Price'] - stats['Off-Peak Hours']['Mean Price']:.2f} JPY/kWh.

## Price Volatility
- The price volatility (standard deviation) is {stats['Overall Statistics']['Std Dev']:.2f} JPY/kWh.
- The price range (max-min) is {stats['Overall Statistics']['Price Range']:.2f} JPY/kWh.

## Recommendations
1. Schedule high-consumption activities during {int(stats['Off-Peak Hours']['Lowest Hour']):02d}:00 when prices are lowest.
2. Avoid peak consumption during {int(stats['Peak Hours (8:00-20:00)']['Peak Hour']):02d}:00 when prices are highest.
3. Consider load shifting from peak to off-peak hours to optimize costs.
"""
        # Save report
        self.generate_report(report, report_name='future_forecast_report.md')
        
        return stats, hourly_stats
    
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

    # 添加经济学分析方法
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
            predictions = self.predict(df[self.feature_columns])
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
        forecaster = JPEXPriceForecastXGB()
        
        print("Loading data...")
        df = forecaster.load_data('spot_summary_2024.csv')
        
        print("Creating features...")
        df = forecaster.create_features(df)
        
        print("Preparing features...")
        X, y = forecaster.prepare_features(df)
        
        print("Splitting data...")
        # Use the last two days of data as the test set
        test_size = 48 * 2  # 2 days, 48 time periods each day
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        X_train_val = X.iloc[:-test_size]
        y_train_val = y.iloc[:-test_size]
        
        # Split the remaining data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        forecaster.train_model(X_train, y_train, X_val, y_val)
        
        print("Making predictions...")
        predictions = forecaster.predict(X_test)
        
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
        
        # Generate detailed report
        print("\nGenerating detailed prediction report...")
        report = forecaster.generate_prediction_report(X_test, y_test, predictions, metrics)
        
        # Predict prices for December 7, 2024
        print("\nGenerating predictions for December 7, 2024...")
        future_predictions = forecaster.predict_specific_date(df)
        stats, hourly_stats = forecaster.analyze_future_predictions(future_predictions)
        
        print("\nKey findings for December 7, 2024:")
        print(f"- Average predicted price: {stats['Overall Statistics']['Mean Price']:.2f} JPY/kWh")
        print(f"- Peak hour price ({int(stats['Peak Hours (8:00-20:00)']['Peak Hour']):02d}:00): {stats['Peak Hours (8:00-20:00)']['Max Price']:.2f} JPY/kWh")
        print(f"- Lowest price hour ({int(stats['Off-Peak Hours']['Lowest Hour']):02d}:00): {stats['Off-Peak Hours']['Min Price']:.2f} JPY/kWh")
        
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
        
        print("\nAll results have been saved to the 'outputs_XGB' directory.")
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'hourly_metrics': hourly_metrics,
            'report': report,
            'future_predictions': future_predictions,
            'future_stats': stats,
            'elasticity_df': elasticity_df,
            'volatility_df': volatility_df,
            'sensitivity_results': sensitivity_results
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
