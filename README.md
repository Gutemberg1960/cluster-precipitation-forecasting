
# Cluster-Based Seasonal Precipitation Forecasting

This project presents a methodology to forecast monthly precipitation over the Paraíba do Sul River Basin using cluster-based modeling and uncertainty estimation. The forecasting approach is based on machine learning (ML) regressors trained separately for each cluster of climatologically similar points.

## Project Structure

All files are located in the **root directory** of this repository:

- `train_models_by_cluster.py`: Trains multiple ML models per cluster, performs bias correction, and saves evaluation metrics and figures.
- `estimate_uncertainty_by_cluster.py`: Estimates forecast uncertainty for each cluster using a secondary model to predict the squared training residuals.
- `plot_test_residuals_distribution.py`: Plots the distribution of test residuals with a Gaussian fit per cluster.
- `glob_1993_2023.xlsx`: Full historical dataset (1993–2023) with predictor variables and target monthly precipitation, already clustered.

## How to Run the Scripts

1. **Install the required Python packages**:

```bash
pip install pandas numpy matplotlib openpyxl scikit-learn xgboost lightgbm scipy
```

2. **Run model training and evaluation**:

```bash
python train_models_by_cluster.py
```

3. **Estimate forecast uncertainty per cluster**:

```bash
python estimate_uncertainty_by_cluster.py
```

4. **Plot residual histograms for each cluster**:

```bash
python plot_test_residuals_distribution.py
```

## Dataset Access

The file `glob_1993_2023.xlsx` contains the full dataset (including cluster assignments). If you encounter download/access issues, contact us for an alternative:

- Gutemberg Borges França – gutemberg@lma.ufrj.br  
- Vinícius Albuquerque de Almeida – vinicius@lma.ufrj.br  

Or use this link for alternate download:  
[Google Sheets Dataset](https://docs.google.com/spreadsheets/d/1IoyKQdPh0c8k3GqzhwiZh1vucIjmvi5j/edit?usp=drive_link)

## Authors

Gutemberg Borges França  
Vinícius Albuquerque de Almeida  
Applied Meteorology Laboratory, Department of Meteorology, Institute of Geosciences, Federal University of Rio de Janeiro (UFRJ)

---

This repository demonstrates a fully functional, reproducible framework for regional precipitation forecasting using unsupervised clustering and robust ML modeling techniques.
