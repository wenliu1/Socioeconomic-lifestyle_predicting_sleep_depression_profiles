# **Socioeconomic and lifestyle factors predict the association between sleep health and depression**

This repository contains the analysis code for the manuscript "Socioeconomic and lifestyle factors predict the association between sleep health and depression".

## Repository Structure
```
Main/
│
├── README.md
├── Multivariate_GLM
│   ├── MultivariateGLM_depression.ipynb (Regressed out sleep variables from depression)
│   └── MultivariateGLM_sleep.ipynb   (Regressed out depression variables from sleep variables)
│
├── PCA
│   ├── PCA_SH_CV.ipynb   (PCA for overall sleep health)
│   ├── PCA_depression_CV.ipynb   (PCA for overall depression)
│   ├── PCA_depression_residuals.ipynb   (PCA for the residuals of depression)
│   └── PCA_sleep_residuals.ipynb   (PCA for the residuals of sleep health)
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
└── rCCA
    ├── rCCA.ipynb    (regularised canonical correlation analysis)
```
