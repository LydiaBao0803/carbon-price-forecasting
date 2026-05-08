import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt

# Load prediction and actual data
def load_prediction_data(model_name, base_path, y_min = 0.01, y_max = 100.29):
    y_pred = pd.read_csv(f'{base_path}/{model_name}_y_pred_vs_actual.csv')['y_pred'].values
    y_test = pd.read_csv(f'{base_path}/{model_name}_y_pred_vs_actual.csv')['y_test'].values

    # Rescale data
    y_pred = y_pred * (y_max - y_min) + y_min
    y_test = y_test * (y_max - y_min) + y_min

    return y_pred, y_test

# calculate errors
def calculate_errors(y_true, y_pred):
    errors = y_true - y_pred
    return errors

# Kernel Density Estimation
def kde_estimation(errors, bandwidth='scott'):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(errors[:, None])
    return kde

# Calculate confidence interval
def calculate_confidence_intervals(y_pred, kde_model, alpha=0.9):
    sample_errors = kde_model.sample(10000)
    lower_bound = np.percentile(sample_errors, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sample_errors, (1 + alpha) / 2 * 100)
    lower_interval = y_pred + lower_bound
    upper_interval = y_pred + upper_bound
    return lower_interval, upper_interval, alpha

# Evaluate model
def evaluate_model(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    residuals = calculate_errors(y_test, y_pred)
    return rmse, mae, mape, residuals

# Plot prediction with confidence intervals
def plot_with_confidence_intervals(y_pred, y_test, lower_interval, upper_interval, model_name, alpha):
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Prediction', linestyle='dashed')
    plt.fill_between(np.arange(len(y_pred)), lower_interval, upper_interval,
                     color='gray', alpha=0.5, label='Confidence Interval')
    plt.ylabel('Value')
    plt.xticks([])
    plt.grid(True)
    plt.legend()
    plt.savefig(f'../interval_forcast/{model_name}_prediction_with_confidence_intervals.png')
    plt.show()

def main():
    base_path = '../point_forcast'
    model_name = 'LSTM_freq_factors_horizon_1'
    y_pred, y_test = load_prediction_data(model_name, base_path)
    rmse, mae, mape, residuals = evaluate_model(y_pred, y_test)
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}')
    kde_model = kde_estimation(residuals)
    lower_interval, upper_interval, alpha = calculate_confidence_intervals(y_pred, kde_model)
    plot_with_confidence_intervals(y_pred, y_test, lower_interval, upper_interval, model_name, alpha)

if __name__ == '__main__':
    main()