# Cluster-Based Seasonal Precipitation Forecasting with Uncertainty Estimation

This repository presents a complete workflow for seasonal precipitation forecasting based on clustering techniques and machine learning models. The approach was developed using data from the Paraíba do Sul River Basin (19932023), grouped into climatologically homogeneous zones.

The project includes:
- Clustering of monitoring stations based on monthly precipitation climatology.
- Model training and validation for each cluster using machine learning algorithms.
- Estimation of forecast uncertainty based on residual modeling.

##  Repository Structure


cluster-precipitation-forecasting/

 data/                      # Directory for input data (user must add)
    glob_1993_2023.xlsx    # Monthly dataset with climate variables and cluster labels

 output/                    # Directory for storing model results and plots

 src/                       # Source code scripts
    cluster_analysis.py        # K-means clustering of precipitation patterns
    train_cluster_models.py    # Model training and evaluation by cluster
    estimate_uncertainty.py    # Forecast uncertainty modeling using residuals

 README.md                 # Project description and usage
 requirements.txt          # Required Python packages
 LICENSE


##  Methodology Overview

1. Data Preprocessing:
   - Daily data aggregated to monthly values.
   - Incomplete monthly records removed.
   - Target: precipitation of the following month (1-month lead).

2. Clustering:
   - K-means applied to normalized monthly climatology (19932023).
   - Optimal number of clusters determined using the Elbow method.

3. Model Training:
   - Algorithms: MLP (Deep), HistGradientBoosting, LightGBM, ExtraTrees, XGBoost.
   - Pipeline: StandardScaler + SelectKBest + GridSearchCV.
   - Evaluation metrics: R², RMSE, MAE, Pearson R.

4. Forecast Bias Correction:
   - Systematic bias removed by subtracting mean error from test predictions.

5. Uncertainty Estimation:
   - Residuals from training set used to fit a second model (same configuration).
   - Prediction standard deviation (STD) computed as uncertainty for test set.
   - Distribution of test residuals analyzed with Gaussian fit.

##  How to Use

1. Install dependencies:

bash
pip install -r requirements.txt


2. Add your input data to the data/ folder:
   - File: glob_1993_2023.xlsx

3. Run clustering:

bash
python src/cluster_analysis.py


4. Train models by cluster:

bash
python src/train_cluster_models.py


5. Estimate uncertainty:

bash
python src/estimate_uncertainty.py


##  Outputs

- Model performance tables (with and without bias correction)
- Scatter plots comparing predicted vs. observed precipitation
- Uncertainty statistics: median, minimum, and maximum STD
- Histogram of test residuals with Gaussian fit

##  Authors

- Gutemberg Borges França  
- Vinícius Albuquerque de Almeida  
Laboratory of Applied Meteorology, Department of Meteorology, Institute of Geosciences, Federal University of Rio de Janeiro (UFRJ), Brazil

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

##  Setup and Execution Instructions

### 1. Create and activate a virtual environment

Using conda:
bash
conda create -n cluster_forecast python3.11 -y
conda activate cluster_forecast


Using venv:
bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venvScriptsactivate   # On Windows


### 2. Install required libraries
bash
pip install -r requirements.txt


---

##  Recommended .gitignore for Python projects

Create a .gitignore file in the root of your project with the following content:


# Python cache and temporary files
__pycache__/
.py[cod]
.egg-info/
.log
.tmp

# Virtual environments
env/
venv/
.venv/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Sensitive or large data
.xlsx
.csv
.pkl
.h5

# Output directories
output/
results/


Remove .xlsx if you plan to include data files in the repository.

---

