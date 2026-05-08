import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.spatial.distance import pdist, squareform

# load selected features and data including EUA prices
selected_features = ['EUA', 'CAC40', 'CRB', 'DJI', 'Oil', 'Gas', 'EUA_Vol.', 'EUR_USD', 'FTSE100', 'DAX30', 'NDX100', 'EUA_RV']
data = pd.read_csv('../data_clean.csv')

X_selected = data[selected_features]

# VMD decomposition for each feature and EUA prices
def apply_vmd(signal, alpha=2000, tau=0, K=11, DC=0, init=1, tol=1e-7):
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    return u

# Calculate sample entropy for a time series
def sample_entropy(signal, m=2):
    r = 0.2 * np.std(signal)
    N = len(signal)
    X = np.array([signal[i: i + m] for i in range(N - m + 1)])
    dists = pdist(X, 'chebyshev')
    dists = squareform(dists)
    B = np.sum(dists < r) / (N - m + 1)
    m += 1
    X = np.array([signal[i: i + m] for i in range(N - m + 1)])
    dists = pdist(X, 'chebyshev')
    dists = squareform(dists)
    A = np.sum(dists < r) / (N - m + 1)
    return -np.log(A / B)

# Apply VMD and SE to each feature
decomposed_signals = {}
for feature in selected_features:
    print(f'Decomposing {feature}...')
    signal = X_selected[feature].values
    modes = apply_vmd(signal)

    # Calculate the sample entropy of each mode
    entropies = [sample_entropy(modes[i]) for i in range(modes.shape[0])]
    print(f'Sample entropies for {feature}: {entropies}')
    
    # Categorize modes based on entropy
    high_freq_modes = []
    mid_freq_modes = []
    low_freq_modes = []

    for i, entropy in enumerate(entropies):
        if entropy > 0.25:
            high_freq_modes.append(modes[i])
        elif 0.1 <= entropy <= 0.25:
            mid_freq_modes.append(modes[i])
        else:
            low_freq_modes.append(modes[i])

    # Combine modes within each frequency group
    if high_freq_modes:
        high_freq = np.sum(high_freq_modes, axis=0)
        data[f'{feature}_high_freq'] = high_freq
    
    if mid_freq_modes:
        mid_freq = np.sum(mid_freq_modes, axis=0)
        data[f'{feature}_mid_freq'] = mid_freq
    
    if low_freq_modes:
        low_freq = np.sum(low_freq_modes, axis=0)
        data[f'{feature}_low_freq'] = low_freq

    decomposed_signals[feature] = modes

    # Plot the decomposed modes and their entropy values
    plt.figure(figsize=(6, 10), dpi=200)
    for i in range(modes.shape[0]):
        plt.subplot(modes.shape[0], 1, i + 1)
        plt.plot(modes[i])
        plt.title(f'Mode {i + 1} of {feature} (SE: {entropies[i]:.2f})')
    plt.tight_layout()
    plt.savefig(f'../Decomposed/decomposed_{feature}.png')

# Save the final dataset with high, mid, and low-frequency components
data.to_csv('../data_VMD_freq.csv', index=False)

# plot the different frequency components
high_freq_features = [f'{feature}_high_freq' for feature in selected_features if f'{feature}_high_freq' in data.columns]
mid_freq_features = [f'{feature}_mid_freq' for feature in selected_features if f'{feature}_mid_freq' in data.columns]
low_freq_features = [f'{feature}_low_freq' for feature in selected_features if f'{feature}_low_freq' in data.columns]

def plot_modes(features, title, filename):
    plt.figure(figsize=(15,20), dpi=200)

    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        plt.plot(data[feature])
        plt.title(feature)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.tight_layout()

    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'../Decomposed/{filename}.png')

plot_modes(high_freq_features, 'High Frequency Modes', 'high_freq')
plot_modes(mid_freq_features, 'Mid Frequency Modes', 'mid_freq')
plot_modes(low_freq_features, 'Low Frequency Modes', 'low_freq')
