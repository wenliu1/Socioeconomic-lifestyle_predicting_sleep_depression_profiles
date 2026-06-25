# **Socioeconomic and lifestyle factors predict the association between sleep health and depression**

This repository contains the analysis code for the manuscript "Socioeconomic and lifestyle factors predict the association between sleep health and depression".

## Repository Structure
```
Main/
│
├── README.md
├── Multivariate_GLM
│   ├── MultivariateGLM_depression.ipynb (Depression variables adjusted for sleep variables)
│   └── MultivariateGLM_sleep.ipynb   (Sleep variables adjusted for depression variables)
│
├── PCA
│   ├── PCA_SH_CV.ipynb   (PCA for overall sleep health)
│   ├── PCA_depression_CV.ipynb   (PCA for overall depression)
│   ├── PCA_depression_residuals.ipynb   (PCA of depression-specific variance after removing variance explained by sleep health)
│   └── PCA_sleep_residuals.ipynb   (PCA of sleep-health-specific variance after removing variance explained by depression)
│ 
├── Prediction
│   ├── ElasticNet_GSCV.py   (Predictive model with ElasticNet)
│   ├── Execute_ElasticNet_GSCV.py   (Execute the model)
│   ├── Lasso_GSCV.py   (Predictive model with Lasso)
│   ├── Exercute_Lasso_GSCV.py   (Execute the model)
│   ├── LinearSVR_GSCV.py   (Predictive model with Linear support vector regression)
│   ├── Exercute_LinearSVR_GSCV.py   (Execute the model)
│   ├── Ridge_GSCV.py   (Predictive model with Ridge regression)
│   ├── Execute_Ridge_GSCV.py   (Execute the model)
│   ├── XGboost_Optuna.py   (Predictive model with XGBoost)
│   ├── Execute_XGBoost.py   (Execute the model)
│   ├── RBF_Optuna.py   (Predictive model with support vector regression with radial basis function kernel)
│   ├── Execute_RBF.py   (Execute the model)
│   ├── RF_Optuna.py   (Predictive model with random forest)
│   └── Execute_RF.py   (Execute the model)
│ 
├── Preprocessing
│   ├── Data_preprocessing.py   (Changed variable names and revised the direction to ensure the higher score the more severe symptoms)
│
├── SHAP
│   ├── SHAP.py (Calculate SHAP scores)
│
├── rCCA
│   ├── rCCA.ipynb    (regularised canonical correlation analysis)
│
└── Utils
    ├── confound_removal.py (package for regression out confounds)
    └──UKB_sub_sleep.yml (virtual environment)
```
## Analysis Workflow
### 1. Data Preprocessing
We renamed and flipped the direction of some variables to keep all variables in consistent direction -- higher scores indicates more severe symptoms.

- `Preprocessing/Data_preprocessing.py`
  - Harmonises variable naming
  - Ensures consistent directionality (higher scores indicate more severe symptoms)
- `Utils/confound_removal.py`
  - Regresses out confounding variables prior to downstream analyses

---

### 2. Regularised Canonical Correlation Analysis (rCCA)
Multivariate associations between sleep health and depression were evaluated using rCCA.

- `rCCA/rCCA.ipynb`
  - Regularised canonical correlation analysis between sleep and depression, get the canonical variates of depression as the target in downstream predictive models.

---

### 3. Multivariate Association Analysis (GLM)
For non-sleep-related depression, sleep health variables were adjusted for depression, and for non-depression-related sleep, depression variables were adjusted for sleep health.

- `Multivariate_GLM/MultivariateGLM_depression.ipynb`
  - Depression variables adjusted for sleep variables
- `Multivariate_GLM/MultivariateGLM_sleep.ipynb`
  - Sleep variables adjusted for depression variables

---

### 4. Dimensionality Reduction (PCA)
To derive comparable target profiles aligned with sleep-related depression,  principal component analysis was applied to extract latent components of overall sleep health, overall depression, the residuals of sleep health and depression.

- `PCA/PCA_SH_CV.ipynb`
  - Overall sleep health components
- `PCA/PCA_depression_CV.ipynb`
  - Overall depression components
- `PCA/PCA_depression_residuals.ipynb`
  - Depression-specific variance after removing shared variance with sleep
- `PCA/PCA_sleep_residuals.ipynb`
  - Sleep-specific variance after removing shared variance with depression

---

### 5. Predictive Modelling
Different machine learning models were trained to predict sleep-related and depression-related profiles using lifestyle and socioeconomic variables.

Models include:
- Elastic Net (`ElasticNet_GSCV.py`)
- Lasso (`Lasso_GSCV.py`)
- Ridge regression (`Ridge_GSCV.py`)
- Linear SVR (`LinearSVR_GSCV.py`)
- RBF-SVR (`RBF_Optuna.py`)
- Random Forest (`RF_Optuna.py`)
- XGBoost (`XGBoost_Optuna.py`)

Execution scripts:
- `Execute_*.py` files run model training and evaluation pipelines using nested cross-validation and hyperparameter tuning.

---

### 6. Model Interpretation (SHAP)
Feature contributions to predictive models are quantified using SHAP values.

- `SHAP/SHAP.py`
  - Computes SHAP values for trained models

---

### 7. Environment
All analyses were performed using the environment specified in:

- `Utils/UKB_sub_sleep.yml`

To reproduce the environment:

```bash
conda env create -f Utils/UKB_sub_sleep.yml
conda activate UKB_sub_sleep
```

## Data Acquisition
[UK Biobank](https://www.ukbiobank.ac.uk/)
