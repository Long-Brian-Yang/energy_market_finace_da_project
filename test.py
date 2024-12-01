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

# Additional imports for advanced visualizations
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
import statsmodels.api as sm

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

        # Add 'date' column manually
        df['date'] = '2024-01-01'  # 替换为实际日期
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
        
        # Check if all features exist in df
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"The following required features are missing from the DataFrame: {missing_features}")
        
        self.feature_columns = features
        X = df[features]
        y = df['system_price']
        
        return X, y
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model and record evaluation metrics."""
        self.model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            eval_metric='rmse'
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=100,
        )
        
        # Retrieve evaluation results
        evals_result = self.model.evals_result()
        self.evals_result = evals_result
    
    def plot_training_history(self):
        """Plot training and validation RMSE over iterations."""
        if not hasattr(self, 'evals_result'):
            raise AttributeError("No evaluation results found. Please train the model first.")
        
        epochs = len(self.evals_result['validation_0']['rmse'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, self.evals_result['validation_0']['rmse'], label='Train')
        plt.plot(x_axis, self.evals_result['validation_1']['rmse'], label='Validation')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('RMSE')
        plt.title('XGBoost Training and Validation RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_history.png'))
        plt.close()

        print("Training history plot saved successfully.")
    
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
    
    def plot_shap_values(self, X_train, num_features=20):
        """Plot SHAP summary and dependence plots."""
        # Create SHAP explainer
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_train)

        # SHAP summary plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'shap_feature_importance.png'))
        plt.close()

        # SHAP summary plot (dot)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title('SHAP Feature Impact')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'shap_summary.png'))
        plt.close()

        # SHAP dependence plots for top features
        top_features = self.model.feature_names_in_[:num_features]
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values.values, X_train, show=False)
            plt.title(f'SHAP Dependence Plot for {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'shap_dependence_{feature}.png'))
            plt.close()

        print("SHAP plots saved successfully.")
    
    def plot_partial_dependence(self, X_train, features, grid_resolution=100):
        """Plot Partial Dependence Plots for specified features."""
        plt.figure(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(
            self.model, X_train, features, grid_resolution=grid_resolution, 
            kind='average', ax=plt.gca()
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'partial_dependence.png'))
        plt.close()

        print("Partial Dependence Plot saved successfully.")
    
    def plot_learning_curve(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """Plot learning curves for the model."""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        
        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Error')
        plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation Error')
        plt.xlabel('Training Set Size')
        plt.ylabel('Error (MSE)')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curve.png'))
        plt.close()

        print("Learning Curve plot saved successfully.")
    
    def plot_residuals_autocorrelation(self, y_test, predictions, lags=20):
        """Plot autocorrelation of residuals."""
        residuals = y_test - predictions
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=ax)
        plt.title('Autocorrelation of Residuals')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residuals_autocorrelation.png'))
        plt.close()

        print("Residuals Autocorrelation plot saved successfully.")
    
    def plot_residuals_time_series(self, y_test, predictions):
        """Plot residuals over time."""
        residuals = y_test - predictions
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.index, residuals, label='Residuals', color='purple')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title('Prediction Residuals Over Time')
        plt.xlabel('Time')
        plt.ylabel('Residual (JPY/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residuals_time_series.png'))
        plt.close()

        print("Residuals Time Series plot saved successfully.")
    
    def plot_predicted_actual_distribution(self, y_test, predictions):
        """Plot distribution comparison between predicted and actual values."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_test, label='Actual', shade=True, color='blue')
        sns.kdeplot(predictions, label='Predicted', shade=True, color='orange')
        plt.title('Distribution of Actual vs Predicted Prices')
        plt.xlabel('Price (JPY/kWh)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'predicted_actual_distribution.png'))
        plt.close()

        print("Predicted vs Actual Distribution plot saved successfully.")
    
    def plot_feature_correlation(self, df):
        """Plot correlation matrix heatmap for features."""
        plt.figure(figsize=(16, 14))
        correlation_matrix = df[self.feature_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_correlation_heatmap.png'))
        plt.close()

        print("Feature Correlation Heatmap saved successfully.")
    
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
        plt.tight_layout()
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
        plt.tight_layout()
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
        for section, metrics_section in report.items():
            print(f"\n{section}:")
            for metric, value in metrics_section.items():
                print(f"{metric}: {value:.4f}")
                
        return report

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
        
        # 确认 self.feature_columns 已被正确设置
        print(f"Feature columns: {forecaster.feature_columns}")
        
        # 绘制特征相关性热图
        print("\nPlotting Feature Correlation Heatmap...")
        forecaster.plot_feature_correlation(df)
        
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
        
        # Plot training history
        print("\nPlotting Training History...")
        forecaster.plot_training_history()
        
        # Generate SHAP plots
        print("\nGenerating SHAP plots...")
        forecaster.plot_shap_values(X_train, num_features=20)
        
        # Generate Partial Dependence Plots
        print("\nGenerating Partial Dependence Plots...")
        top_features = forecaster.model.feature_names_in_[:5]  # 选择前5个重要特征
        forecaster.plot_partial_dependence(X_train, features=top_features)
        
        # Generate Learning Curve
        print("\nGenerating Learning Curve...")
        forecaster.plot_learning_curve(X_train_val, y_train_val)
        
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
        
        # Plot residuals autocorrelation
        print("\nPlotting Residuals Autocorrelation...")
        forecaster.plot_residuals_autocorrelation(y_test, predictions)
        
        # Plot residuals time series
        print("\nPlotting Residuals Time Series...")
        forecaster.plot_residuals_time_series(y_test, predictions)
        
        # Plot predicted vs actual distribution
        print("\nPlotting Predicted vs Actual Distribution...")
        forecaster.plot_predicted_actual_distribution(y_test, predictions)
        
        # Generate detailed prediction report
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
        
        print("\nAll results have been saved to the 'outputs_XGB' directory.")
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'hourly_metrics': hourly_metrics,
            'report': report,
            'future_predictions': future_predictions,
            'future_stats': stats
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
