"""
estimate_uncertainty.py
-------------------------
This script estimates the uncertainty of the optimal forecasting model for each cluster
by training a secondary MLP on squared residuals from the training data.
It saves the summary statistics of predicted standard deviation per cluster and plots the histogram.
"""

import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from scipy.stats import norm

data_path = "data/glob_1993a2023_mo_join_filtered_com_clusters.xlsx"
output_path = "outputs/uncertainty"
os.makedirs(output_path, exist_ok=True)

# === Features and model configuration ===
features = ['latitude', 'longitude', 'u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y']

base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_regression, k=8)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 30, 20), activation='relu', alpha=0.001,
                         early_stopping=True, max_iter=3000, random_state=42))
])

# === Load dataset ===
df = pd.read_excel(dataset_path)
df.columns = df.columns.str.replace(" ", "_")

# === Loop over clusters ===
for cluster_id in sorted(df['cluster'].dropna().unique()):
    df_c = df[df['cluster'] == cluster_id]

    # Train/test split
    X_train = df_c[df_c.year_x <= 2018][features]
    Y_train = df_c[df_c.year_x <= 2018]['pr_x']
    X_test = df_c[df_c.year_x > 2018][features]
    Y_test = df_c[df_c.year_x > 2018]['pr_x']

    # First model: forecast precipitation
    forecast_model = clone(base_model)
    forecast_model.fit(X_train, Y_train)
    Y_pred_train = forecast_model.predict(X_train)
    Y_pred_test = forecast_model.predict(X_test)

    # Residuals
    residuals_train = Y_train.values - Y_pred_train
    residuals_test = Y_test.values - Y_pred_test

    # Second model: predict residual² for uncertainty
    uncertainty_model = clone(base_model)
    uncertainty_model.fit(X_train, residuals_train ** 2)
    predicted_var = uncertainty_model.predict(X_test)
    predicted_std = np.sqrt(np.maximum(predicted_var, 0))

    # Save summary
    summary = pd.DataFrame({
        'Cluster': [cluster_id],
        'Max STD': [np.max(predicted_std)],
        'Min STD': [np.min(predicted_std)],
        'Median STD': [np.median(predicted_std)]
    })
    summary.to_excel(output_path / f"uncertainty_cluster_{cluster_id}.xlsx", index=False)

    # Plot: training residuals
    mu_train, std_train = norm.fit(residuals_train)
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_train, bins=25, density=True, alpha=0.6, label="Training Residuals", color='skyblue', edgecolor='black')
    x = np.linspace(*plt.xlim(), 100)
    plt.plot(x, norm.pdf(x, mu_train, std_train), 'r--', label=f'Gaussian Fit\\nμ={mu_train:.2f}, σ={std_train:.2f}')
    plt.title(f"Training Residuals - Cluster {cluster_id}")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / f"train_residuals_cluster_{cluster_id}.png", dpi=600)
    plt.show()

    # Plot: test residuals
    mu_test, std_test = norm.fit(residuals_test)
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_test, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label="Test Residuals")
    x = np.linspace(*plt.xlim(), 100)
    plt.plot(x, norm.pdf(x, mu_test, std_test), 'r--', linewidth=2, label=f'Gaussian Fit\\nμ={mu_test:.2f}, σ={std_test:.2f}')
    plt.title(f"Test Residuals with Gaussian Fit - Cluster {cluster_id}")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / f"test_residuals_cluster_{cluster_id}.png", dpi=600)
    plt.show()

print(" Process completed for all clusters. Results saved to:", output_path)
