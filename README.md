
# Cluster-Based Seasonal Precipitation Forecasting

This repository provides a structured framework for seasonal precipitation forecasting over the Paraíba do Sul River Basin using machine learning. The methodology includes K-means clustering, model optimization per cluster, and uncertainty estimation.

## Project Structure

```
cluster-precipitation-forecasting/
├── data/
│   └── sample_glob_1993_2023.xlsx      # Sample of the dataset for testing purposes only
├── outputs/                            # Output folder for plots and result tables
├── src/
│   ├── cluster_analysis_train.py       # (Optional) Script to perform or validate clustering
│   ├── train_forecasting_models.py     # Trains ML models per cluster and saves evaluation metrics
│   └── estimate_uncertainty.py         # Uses the optimal model to estimate forecast uncertainty
└── README.md                           # Project overview, usage, and authorship
```

## How to Run

### 1. Install Required Libraries

Use Python 3.8+ and install dependencies:

```bash
pip install -r requirements.txt
```

You will need:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `lightgbm`
- `openpyxl`

### 2. Run Model Training

```bash
python src/train_forecasting_models.py
```

### 3. Run Uncertainty Estimation

```bash
python src/estimate_uncertainty.py
```

## Dataset Access

A sample of the dataset (`sample_glob_1993_2023.xlsx`) is provided for testing.

To access the full dataset, please contact:

**Gutemberg Borges França** or **Vinícius Albuquerque de Almeida**  
Applied Meteorology Laboratory – UFRJ  
📧 Email: gutemberg@lma.ufrj.br or vinicius@lma.ufrj.br  
🔗 Or download directly from:  
[Google Sheets Link](https://docs.google.com/spreadsheets/d/1IoyKQdPh0c8k3GqzhwiZh1vucIjmvi5j/edit?usp=drive_link&ouid=102911351625129185133&rtpof=true&sd=true)

## Authors

Gutemberg Borges França  
Vinícius Albuquerque de Almeida

Applied Meteorology Laboratory  
Department of Meteorology  
Institute of Geosciences  
Federal University of Rio de Janeiro (UFRJ)
