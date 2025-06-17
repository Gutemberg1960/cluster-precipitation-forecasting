
# Cluster-Based Seasonal Precipitation Forecasting

This repository contains Python routines for forecasting seasonal precipitation in the Paraíba do Sul River Basin using machine learning models trained by precipitation clusters.

## Structure

- `src/` — Python scripts for training, uncertainty estimation, and residual analysis.
- `data/` — A sample dataset for demonstration purposes.

## Scripts

Each script is placed in `src/` and fully documented in English:

- `train_models_by_cluster.py`: Trains several ML models (MLP, XGBoost, etc.) per cluster, with bias correction and metrics output.
- `estimate_uncertainty_by_cluster.py`: Models the squared residuals of the training set to estimate prediction uncertainty.
- `plot_test_residuals_distribution.py`: Plots the distribution of test residuals and overlays a fitted Gaussian curve.

## Data Access

A **sample dataset** is included in `data/sample_glob_1993_2023_strict.xlsx`.  
To access the full dataset (`glob_1993_2023.xlsx`), please contact:

- Gutemberg Borges França – gutemberg@lma.ufrj.br  
- Vinícius Albuquerque de Almeida – vinicius@lma.ufrj.br  

Or download directly from this [Google Sheets link](https://docs.google.com/spreadsheets/d/1IoyKQdPh0c8k3GqzhwiZh1vucIjmvi5j).

## Authors

Gutemberg Borges França and Vinícius Albuquerque de Almeida  
Laboratório de Meteorologia Aplicada, Departamento de Meteorologia, Instituto de Geociência, Universidade Federal do Rio de Janeiro (UFRJ)
