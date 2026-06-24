#################################################
# The script only uses sex and age as confounds #
#################################################

import sys
sys.path.append('working_path')

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from confound_removal import ConfoundRemover
from XGboost_Optuna import XGBoptuna

from joblib import Parallel, delayed
import time
import os

"""
The target has been preprocessed and deconfounded in the CCA script before generating the transformed scores.
The direction of all targets should be flipped.
"""

# Define variables (outside loop - same for all folds)
"""
Define variables
"""
# Define demographical variables
Age = ['age_variable']
Sex = ['Sex']  
Economy = ['economy_variable1']  

# Define depression variables
# Trans_N_12 and Trans_RDS_4 are the transformed scores from the CCA analysis, which are used as targets for ML.
Trans_N_12 = ['Trans_n_12_depression_variables']
Trans_RDS_4 = ['Trans_rds_4_depression_variables']

# Define lifestyle variables
Lifestyle_alco = ['alcohol_variables']
Lifestyle_smok = ['smoke_variables']
Lifestyle_diet = ['diet_variables']
Lifestyle_acti = ['activity_variables'] 
Lifestyle_electro = ['electronic_variables']
Lifestyle_sun = ['sun_variables']
Employment_status = ['employment_variables']
Social_support = ['social_support_variables']

Lifestyle_features = Lifestyle_alco + Lifestyle_smok + Lifestyle_diet + Lifestyle_acti + Lifestyle_electro + Lifestyle_sun + Employment_status + Social_support

Demo_lifestyle_features = Lifestyle_features + Economy # Combine all features for preprocessing

confounds = Age  # Sex not included for normalization 
joined_confounds_list = confounds + Sex
all_columns = Demo_lifestyle_features + joined_confounds_list 

# Separate categorical and continuous variables in features for different standardization
categorical_features = [col for group in [Lifestyle_alco, Lifestyle_smok, Employment_status] for col in group]

# Define nested cross-validation
n_splits_outer = 5
n_repeats_outer = 2
n_splits_inner = 5
n_repeats_inner = 1
random_seed = 42

Inner_cv = RepeatedKFold(n_repeats=n_repeats_inner, n_splits=n_splits_inner, random_state=random_seed)


# Define functions before the loop
def cross_validation(Features, target, joined_confounds, train_idx, test_idx):
    """
    Prepare training and validation data by stacking X_data, y_data with confounds.
    """
    # Get features and confounds as numpy arrays
    Features_train_noconfound = Features.iloc[train_idx].values
    Features_test_noconfound = Features.iloc[test_idx].values
    confounds_train = joined_confounds.iloc[train_idx].values
    confounds_test = joined_confounds.iloc[test_idx].values

    # Combine features with confounds
    Features_train = np.hstack([Features_train_noconfound, confounds_train])
    Features_test = np.hstack([Features_test_noconfound, confounds_test])

    # Target data has been standardized and confound removal in CCA analysis
    target_train = target[train_idx].values
    target_test = target[test_idx].values

    # Extract eid from test set
    eid_test = eid.iloc[test_idx].values

    return Features_train, target_train, Features_test, target_test, eid_test


def execute_ml_pipeline(Features, target, joined_confounds, models_dict, Outer_cv, Inner_cv, fold, parallel_outer=False, parallel_inner=True):
    """
    Execute the entire ML pipeline for a single fold.
    """
    all_results = []
    start = time.time()

    # Resource allocation logic
    total_cores = os.cpu_count()
    if parallel_inner:
        optuna_n_jobs = total_cores - 1
        outer_n_jobs = 1
    else:
        optuna_n_jobs = 1
        outer_n_jobs = total_cores // 2
    
    # Common save paths
    metrics_path = f'save_path_Fold_{fold}/model_performance.csv'
    predictions_path = f'save_path_Fold_{fold}/model_predictions.csv'
    pipeline_save_path = f'save_path_Fold_{fold}/'
    fold_index_dir = pipeline_save_path

    # Ensure directories exist
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    
    # Function to process a single fold
    def process_fold(fold_info):
        i, (train_idx, test_idx) = fold_info
        Features_train, target_train, Features_test, target_test, eid_test = cross_validation(
            Features, target, joined_confounds, train_idx, test_idx
        )

        # Call XGBoptuna_tuning method
        results = lm_xgb.XGBoptuna_tuning(
            pipeline, Features_train, target_train, Features_test, target_test, 
            models_dict, Inner_cv, i, 
            save_path=metrics_path,
            pipeline_save_path=pipeline_save_path,
            optuna_n_jobs=optuna_n_jobs,
            n_jobs=1, 
            eid_test=eid_test
        )
        return results

    # Save the fold indices
    fold_info = []
    for i, (train_idx, test_idx) in enumerate(Outer_cv.split(Features)):
        np.save(os.path.join(fold_index_dir, f'fold_{i+1}_test_idx.npy'), test_idx)
        np.save(os.path.join(fold_index_dir, f'fold_{i+1}_train_idx.npy'), train_idx)
        fold_info.append((i, (train_idx, test_idx)))

    # Process folds in parallel or sequential
    if parallel_outer:
        all_results = Parallel(n_jobs=outer_n_jobs)(
            delayed(process_fold)(fold_info_item)
            for fold_info_item in fold_info
        )
    else:
        all_results = [process_fold(fold_item) for fold_item in fold_info]

    # Combine all results
    results_list = []
    for result in all_results:
        if isinstance(result, pd.DataFrame):
            results_list.append(result)
        elif isinstance(result, list):
            results_list.extend(result)
    
    if not results_list:
        raise ValueError("No valid results were generated")
    
    results_df = pd.concat(results_list, ignore_index=True)

    print(f"Total pipeline execution time for fold {fold}: {time.time() - start:.2f} seconds")
    
    return results_df


# Cross-sectional data were split into 5 folds from cross-validation in rCCA step. In each iteration, one fold was used for rCCA and the other four folds were used for ML.
folds = ['1', '2', '3', '4', '5']

# Loop through each fold
for fold in folds:
    
    # Load data
    df_ML = pd.read_csv(f'working_path/ML_cross_fold_{fold}.csv', low_memory=False)
    
    # Extract features and confounds
    Features = df_ML[Demo_lifestyle_features]
    joined_confounds = df_ML[joined_confounds_list]
    
    # Create indices for the ColumnTransformer
    continuous_features = [var for var in Demo_lifestyle_features if var not in categorical_features]
    cont_features_idx = [all_columns.index(col) for col in continuous_features]
    cat_features_idx = [all_columns.index(col) for col in categorical_features]
    confounds_idx = [all_columns.index(col) for col in confounds]
    sex_idx = [all_columns.index(Sex[0])]

    # Combine both lists of column names
    Trans_columns = Trans_N_12 + Trans_RDS_4  

    # Calculate the target
    df_ML['sum_SRD'] = df_ML[Trans_columns].sum(axis=1)
    target_orig = df_ML['sum_SRD']
    target = -target_orig  # Flip direction of target
    eid = df_ML['eid']
    
    # Define nested cross-validation for this fold
    Outer_cv = RepeatedKFold(n_repeats=n_repeats_outer, n_splits=n_splits_outer, random_state=random_seed)

    # Pipeline setup
    preprocessor_X = ColumnTransformer(
        transformers=[
            ('cont_features', StandardScaler(), cont_features_idx),  # Scale continuous variables
            ('cat_features', 'passthrough', cat_features_idx),      # Skip categorical variables
            ('confounds', StandardScaler(), confounds_idx),         # Scale confounds
            ('sex', 'passthrough', sex_idx)                        # Sex variable
        ],
        sparse_threshold=0
    )

    # Create a pipeline for preprocessing and confounds removal
    pipeline = Pipeline([
        ('preprocessor', preprocessor_X),  
        ('deconfound', ConfoundRemover(n_confounds=joined_confounds.shape[1]))
    ])

    # Initialize model
    lm_xgb = XGBoptuna()
    models_dict = lm_xgb.models_dict

    # Execute pipeline for this fold
    execute_ml_pipeline(Features, target, joined_confounds, models_dict, Outer_cv, Inner_cv, fold, 
                       parallel_outer=False, parallel_inner=True)
    
    print(f"\nCompleted fold {fold}\n")