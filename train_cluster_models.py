"""
train_cluster_models.py
-------------------------
This script trains machine learning models to predict monthly precipitation by cluster.
It performs GridSearchCV to optimize hyperparameters, applies bias correction, and saves metrics and scatter plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

data_path = "data/glob_1993a2023_mo_join_filtered_com_clusters.xlsx"
output_dir = Path("outputs/models")
output_dir.mkdir(parents=True, exist_ok=True)

features = ['latitude', 'longitude', 'u2_y', 'tmin_y', 'tmax_y', 'rs_y', 'rh_y', 'eto_y']

df = pd.read_excel(data_path).dropna()
df.columns = df.columns.str.replace(" ", "_")

models = {
    'MLP': (MLPRegressor(max_iter=5000, early_stopping=True, random_state=42), {
        'model__hidden_layer_sizes': [(50, 30, 20)],
        'model__activation': ['relu'],
        'model__alpha': [0.001]
    }),
    'HistGB': (HistGradientBoostingRegressor(random_state=42), {
        'model__learning_rate': [0.05],
        'model__max_iter': [300],
        'model__max_depth': [7]
    }),
    'ExtraTrees': (ExtraTreesRegressor(random_state=42), {
        'model__n_estimators': [300],
        'model__max_depth': [10]
    }),
    'LightGBM': (LGBMRegressor(random_state=42), {
        'model__n_estimators': [300],
        'model__learning_rate': [0.05],
        'model__max_depth': [5]
    }),
    'XGBoost': (XGBRegressor(objective='reg:squarederror', random_state=42), {
        'model__n_estimators': [300],
        'model__learning_rate': [0.03],
        'model__max_depth': [6]
    })
}

for cluster_id in sorted(df['cluster'].dropna().unique()):
    print(f"Processing Cluster {cluster_id}")
    df_c = df[df['cluster'] == cluster_id]

    X_train = df_c[df_c.year_x <= 2018][features]
    Y_train = df_c[df_c.year_x <= 2018]['pr_x']
    X_test = df_c[df_c.year_x > 2018][features]
    Y_test = df_c[df_c.year_x > 2018]['pr_x']

    for model_name, (regressor, param_grid) in models.items():
        steps = [
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=f_regression)),
            ('model', regressor)
        ]
        pipeline = Pipeline(steps)
        grid = GridSearchCV(pipeline, {
            'select__k': list(range(1, len(features) + 1)),
            **param_grid
        }, scoring='r2', cv=5, n_jobs=-1)

        grid.fit(X_train, Y_train)
        model = grid.best_estimator_
        Y_pred = model.predict(X_test)
        bias = Y_pred.mean() - Y_test.mean()
        Y_pred_corr = Y_pred - bias

        results = {
            'R2': r2_score(Y_test, Y_pred_corr),
            'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred_corr)),
            'MAE': mean_absolute_error(Y_test, Y_pred_corr),
            'Pearson R': pearsonr(Y_test, Y_pred_corr)[0],
            'Bias': bias
        }

        pd.DataFrame([results]).to_excel(
            output_dir / f"metrics_cluster{cluster_id}_{model_name}.xlsx",
            index=False
        )

        plt.figure()
        plt.scatter(Y_test, Y_pred_corr, alpha=0.6)
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
        plt.title(f"Cluster {cluster_id} - {model_name}")
        plt.xlabel("Observed")
        plt.ylabel("Predicted (bias-corrected)")
        plt.grid(True)
        plt.savefig(output_dir / f"scatter_cluster{cluster_id}_{model_name}.png")
        plt.close()
