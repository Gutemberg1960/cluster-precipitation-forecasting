
# Cluster-based Seasonal Precipitation Forecasting

This repository contains Python routines for forecasting monthly precipitation in the Paraíba do Sul River Basin using machine learning models, applied separately to each climatologically homogeneous cluster.

## 🔍 Objective

To identify optimal models per cluster using machine learning regression algorithms and estimate uncertainty for seasonal precipitation predictions.

## 📁 Project Structure

```
cluster-precipitation-forecasting/
│
├── data/                      # Input dataset (sample provided)
│   └── glob_1993_2023.xlsx
│
├── outputs/                   # Output results (graphs, Excel reports)
│
├── src/                       # Source code scripts
│   ├── clustering_analysis.py
│   ├── train_cluster_models.py
│   └── estimate_uncertainty.py
│
└── README.md
```

## 📊 Dataset

The dataset `glob_1993_2023.xlsx` is a stratified sample (up to 250 records per cluster) extracted from the BR-DWGD database. It includes meteorological predictors and target precipitation values for monthly forecasting (1993–2023).

## 👥 Authors

- Gutemberg Borges França  
- Vinícius Albuquerque de Almeida

**Meteorological Applications Lab, Department of Meteorology, Institute of Geosciences, Federal University of Rio de Janeiro (UFRJ)**

