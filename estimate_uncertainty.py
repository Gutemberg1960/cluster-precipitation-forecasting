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

features = ['latitude', 'longitude', 'u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y']
model = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_regression, k=8)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50, 30, 20), activation='relu', alpha=0.001,
                         early_stopping=True, max_iter=3000, random_state=42))
])

df = pd.read_excel(data_path)
df.columns = df.columns.str.replace(" ", "_")

for cluster_id in sorted(df['cluster'].dropna().unique()):
    df_c = df[df['cluster'] == cluster_id]

    X_train = df_c[df_c.year_x <= 2018][features]
    y_train = df_c[df_c.year_x <= 2018]['pr_x']
    y_pred = model.fit(X_train, y_train).predict(X_train)
    residuals_sq = (y_pred - y_train) ** 2

    # Train model on residuals²
    model.fit(X_train, residuals_sq)

    # Predict on test
    X_test = df_c[df_c.year_x > 2018][features]
    y_uncert = np.sqrt(model.predict(X_test))

    summary = pd.DataFrame({
        'Cluster': cluster_id,
        'Max STD': [np.max(y_uncert)],
        'Min STD': [np.min(y_uncert)],
        'Median STD': [np.median(y_uncert)]
    })
    summary.to_excel(f"{output_path}/uncertainty_cluster_{cluster_id}.xlsx", index=False)

    # Plot histogram of training residuals with Gaussian
    plt.figure()
    residuals = y_pred - y_train
    mu, std = norm.fit(residuals)
    plt.hist(residuals, bins=20, density=True, alpha=0.6, label='Residuals')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, norm.pdf(x, mu, std), 'r--', label='Gaussian Fit')
    plt.title(f"Residual Distribution - Cluster {cluster_id}")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/residuals_histogram_cluster_{cluster_id}.png")
    plt.close()

    # Plot histogram of TEST residuals with Gaussian fit
    test_residuals = Y_test.values - corrected_preds
    mu, std = norm.fit(test_residuals)
    plt.figure(figsize=(8, 6))
    plt.hist(test_residuals, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label="Test Residuals")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian Fit\nμ={mu:.2f}, σ={std:.2f}')
    plt.title(f"Test Residuals with Gaussian Fit - Cluster {cluster_id}")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/hist_residuals_cluster_{cluster_id}.png", dpi=300)
    plt.close()
