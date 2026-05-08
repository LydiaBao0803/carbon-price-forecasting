import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Attention, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# -----------Load and preprocess data-----------
# Load and preprocess data
def load_data(file_path, strategy):
    data = pd.read_csv(file_path)

    if strategy == 'freq_factors':
        # Use all freq-related columns as features
        X = data[[col for col in data.columns if 'freq' in col]]
    elif strategy == 'original_factors':
        # Use original factors (without freq)
        X = data[['CAC40', 'CRB', 'DJI', 'Oil', 'Gas', 'EUA_Vol.', 'EUR_USD', 'FTSE100', 'DAX30', 'NDX100', 'EUA_RV']]
    else:
        raise ValueError("Invalid strategy. Choose from 'freq_factors', 'original_factors', 'no_factors'.")

    # Target variable is always EUA
    y = data['EUA']

    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y

# -----------Sliding window-----------
# Create sliding windows
def create_sliding_window(X, y, window_size, forecast_horizon):
    X_window, y_window = [], []
    for i in range(len(X) - window_size - forecast_horizon):
        # Append the features for the current window
        X_window.append(X[i:i+window_size])

        # Append the target value for the specified horizon
        y_window.append(y[i+window_size+forecast_horizon])

    return np.array(X_window), np.array(y_window).reshape(-1, 1)  # Ensure y has the shape (samples, 1)

# Modify the sliding window function for traditional models (2D format)
def create_sliding_window_2d(X, y, window_size, forecast_horizon):
    X_window, y_window = [], []
    for i in range(len(X) - window_size - forecast_horizon):
        X_window.append(X[i:i+window_size].flatten())
        y_window.append(y[i+window_size+forecast_horizon])
    return np.array(X_window), np.array(y_window).reshape(-1, 1)

# -----------Model building-----------
# Build SVM model
def build_svm():
    model = SVR(kernel='rbf')
    return model

# Build KNN model 
def build_knn():
    model = KNeighborsRegressor(n_neighbors=5)
    return model

# Build BPNN model
def build_bpnn():
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
    return model

# Build LSTM with Attention
def build_lstm(input_shape, output_size):
    inputs = Input(shape=input_shape)

    # LSTM layer, return the sequence (time steps)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    
    # Attention layer, ensure the inputs are 3D (batch_size, timesteps, features)
    query = lstm_out
    value = lstm_out
    attention_out = Attention()([query, value])

    # Flatten the attention output to 2D (batch_size, features)
    flattened = tf.keras.layers.Flatten()(attention_out)
    
    # Fully connected output layer
    outputs = Dense(output_size)(flattened)

    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build GRU model
def build_gru(input_shape, output_size):
    inputs = Input(shape=input_shape)
    gru_out = GRU(64, return_sequences=False)(inputs)
    outputs = Dense(output_size)(gru_out)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# -----------Evaluation-----------
# Evaluate function for traditional models
def evaluate_traditional_model(model, X_train, y_train, X_test, y_test, scaler_y, model_name):
    # Train model
    model.fit(X_train, y_train.ravel())

    # Predict
    y_pred = model.predict(X_test)
    
    # Reshape y_pred to 2D before inverse transforming
    y_pred = y_pred.reshape(-1, 1)

    # Inverse scale predictions and true values
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

    # Save predictions and true values to a CSV file
    result_df = pd.DataFrame({'y_pred': y_pred_rescaled.flatten(), 'y_test': y_test_rescaled.flatten()})
    result_df.to_csv(f'../point_forcast/{model_name}_y_pred_vs_actual.csv', index=False)

    return rmse, mae, mape


# Evaluate models and save metrics
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
    
    # Predict
    y_pred = model.predict(X_test)

    # Inverse scale predictions and true values
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

    # Save predictions and true values to a CSV file
    result_df = pd.DataFrame({'y_pred': y_pred_rescaled.flatten(), 'y_test': y_test_rescaled.flatten()})
    result_df.to_csv(f'../point_forcast/{model_name}_y_pred_vs_actual.csv', index=False)

    return rmse, mae, mape

# -----------Model comparison-----------
# Main comparison function
def model_comparison(data_filepath, strategy, window_size=5, forecast_horizon=1):
    # Load data
    X, y, scaler_X, scaler_y = load_data(data_filepath, strategy)

    # Create sliding windows
    X_window, y_window = create_sliding_window(X, y, window_size, forecast_horizon)
    X_train_2d, y_train_2d = create_sliding_window_2d(X, y, window_size, forecast_horizon)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.2, shuffle=False)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_train_2d, y_train_2d, test_size=0.2, shuffle=False)

    # Reshape for LSTM/GRU - samples, timesteps, features    
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Initialize a list to store results for each model
    results_list = []

    # LSTM Model
    lstm_model = build_lstm(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), output_size=1)
    rmse, mae, mape = evaluate_model(lstm_model, X_train_lstm, y_train, X_test_lstm, y_test, scaler_y, f'LSTM_{strategy}_horizon_{forecast_horizon}')
    results_list.append({'Model': f'LSTM_{strategy}_horizon_{forecast_horizon}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    # GRU Model
    gru_model = build_gru(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), output_size=1)
    rmse, mae, mape = evaluate_model(gru_model, X_train_lstm, y_train, X_test_lstm, y_test, scaler_y, f'GRU_{strategy}_horizon_{forecast_horizon}')
    results_list.append({'Model': f'GRU_{strategy}_horizon_{forecast_horizon}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    # SVM Model
    svm_model = build_svm()
    rmse, mae, mape = evaluate_traditional_model(svm_model, X_train_2d, y_train_2d, X_test_2d, y_test_2d, scaler_y, f'SVM_{strategy}_horizon_{forecast_horizon}')
    results_list.append({'Model': f'SVM_{strategy}_horizon_{forecast_horizon}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    # KNN Model
    knn_model = build_knn()
    rmse, mae, mape = evaluate_traditional_model(knn_model, X_train_2d, y_train_2d, X_test_2d, y_test_2d, scaler_y, f'KNN_{strategy}_horizon_{forecast_horizon}')
    results_list.append({'Model': f'KNN_{strategy}_horizon_{forecast_horizon}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    # BPNN Model
    bpnn_model = build_bpnn()
    rmse, mae, mape = evaluate_traditional_model(bpnn_model, X_train_2d, y_train_2d, X_test_2d, y_test_2d, scaler_y, f'BPNN_{strategy}_horizon_{forecast_horizon}')
    results_list.append({'Model': f'BPNN_{strategy}_horizon_{forecast_horizon}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df

# -----------Main function-----------
# Run the comparison with different strategies and forecast horizons
if __name__ == '__main__':
    data_filepath = '../data_VMD_freq.csv'

    # Define the strategies and forecast horizons
    strategies = ['freq_factors', 'original_factors']
    forecast_horizons = [1, 15, 30]

    all_results = []
    for strategy in strategies:
        for horizon in forecast_horizons:
            print(f"Running model with strategy: {strategy} and forecast horizon: {horizon}")
            results = model_comparison(data_filepath, strategy, forecast_horizon=horizon)
            all_results.append(results)

    # Concatenate results for all strategies and horizons and save to a single CSV file
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv('../point_forcast/point_forcast_final_results.csv', index=False)

    # Print the final results
    print(final_results)