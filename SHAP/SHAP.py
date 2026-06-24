#==========================================
# Calculate SHAP values for the test set 
#==========================================

import sys
sys.path.append('working_path')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
import joblib
from joblib import Parallel, delayed

# Initialize a dictionary to store the dataframes
dataframes = {}

# Loop through all 5 files split in rCCA step
for i in range(1, 6):
    # Construct the file path
    file_path = f'working_path/ML_cross_fold_{i}.csv'
    
    # Read the CSV and store in the dictionary
    dataframes[f'df{i}'] = pd.read_csv(file_path)

globals().update(dataframes)

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

shap_values_test_list = []
X_all_test_list = []
feature_names_list = []
eid_test_list = []

def process_fold(df, df_num, fold_num):
    # Get the same inputs from the pipeline (features + confounds)
    Features_full = df[all_columns]
    model = joblib.load(f'working_path/Fold_{df_num}/fold_{fold_num}_pipeline_CV.joblib')
    train_idx = np.load(f'working_path/Fold_{df_num}/fold_{fold_num}_train_idx.npy')
    test_idx = np.load(f'working_path/Fold_{df_num}/fold_{fold_num}_test_idx.npy')

    if isinstance(model, Pipeline):
        # Seperate steps from pipeline 
        preprocessor = model.named_steps['preprocessor']
        deconfounder = model.named_steps['deconfound']
        final_model = model.named_steps['model']

        # Transform input following the steps of the pipeline
        X_transformed = preprocessor.transform(Features_full.values)
        X_deconfounded = deconfounder.transform(X_transformed)

        # Split the deconfounded data into train and test sets
        X_train = X_deconfounded[train_idx]
        X_test = X_deconfounded[test_idx]

        # Extract the eid for each fold
        eids = df['eid']
        eids_test = eids.iloc[test_idx].reset_index(drop=True)
        
        # Get SHAP values for the model
        explainer = shap.TreeExplainer(final_model, data = X_train)
        shap_test_values = explainer.shap_values(X_test)

        # Get all feature names in the original order (after ColumnTransformer)
        transformers = preprocessor.transformers_
        cont_indices = transformers[0][2]    # Continuous features (scaled)
        cat_indices = transformers[1][2]     # Categorical features (passthrough)
        confound_indices = transformers[2][2]  # Confounds (age, scaled)
        sex_indices = transformers[3][2]     # Sex (passthrough)

        # Reconstruct full feature names (order matches ColumnTransformer output)
        all_feature_names = (
            [all_columns[i] for i in cont_indices] +    # Continuous features
            [all_columns[i] for i in cat_indices] +    # Categorical features
            [all_columns[i] for i in confound_indices] + # Confounds (age)
            [all_columns[i] for i in sex_indices]       # Sex
        )

        # Remove confounds to match X_deconfounded
        final_feature_names = [f for f in all_feature_names if f not in joined_confounds_list]

        return shap_test_values, X_test, eids_test, final_feature_names
    
# Process all folds in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_fold)(dataframes[f'df{df_num}'], df_num, fold_num)
    for df_num in range(1, 6)
    for fold_num in range (1, 11)
)

# Collecting results
for res in results:
    if res is not None:
        shap_test_values, X_test, eids_test, final_feature_names = res
        shap_values_test_list.append(shap_test_values)
        X_all_test_list.append(X_test)
        eid_test_list.append(eids_test)
        feature_names_list.append(final_feature_names)

# Stack vertically along rows 
all_shap_test_values = np.vstack(shap_values_test_list)
X_all_test = np.vstack(X_all_test_list) 
all_eids_test = np.concatenate(eid_test_list)

# Convert SHAP values to dataframe and average over eid
df_shap_test = pd.DataFrame(all_shap_test_values, columns=final_feature_names)
df_X_test = pd.DataFrame(X_all_test, columns=final_feature_names)

# Add eid for the corresponding dataframes
df_shap_test['eid'] = all_eids_test
df_X_test['eid'] = all_eids_test

# Compute the mean of shap values for the test set by 
mean_shap_test = df_shap_test.groupby('eid').mean()
mean_X_all_test = df_X_test.groupby('eid').mean()

# Save the SHAP value for each individual
mean_shap_test.to_csv('working_path/individual_SHAP_value_GP_cross.csv')

# Compute mean of absolute SHAP values across all participants in the test set
mean_abs_shap = np.mean(np.abs(mean_shap_test), axis=0)
print(f'The mean value for |SHAP| in test set for XGBoost: {mean_abs_shap}')

# Plot feature importance
shap.summary_plot(
    mean_shap_test, 
    mean_X_all_test, 
    feature_names=final_feature_names,
    plot_type='bar',   
    show=False
)

# Set a figure size before plotting
plt.gcf().set_size_inches(12, 8)

plt.title("Mean Absolute SHAP Values in the test set for XGBoost\n(Feature Importance)", fontsize=16)
plt.xlabel("Mean(|SHAP value|) over all test folds", fontsize=14)
plt.tight_layout()
plt.savefig('working_path/figure_name.png', bbox_inches='tight')
plt.show()
plt.close()

# Create the figure
plt.figure(figsize=(14, 10), dpi=900)

# Generate the SHAP summary plot
shap.summary_plot(
    mean_shap_test.values, 
    mean_X_all_test.values, 
    feature_names=final_feature_names,
    plot_size=None,  
    show=False       
)

# Apply custom styling
ax = plt.gca()

# Title (centered, padded)
ax.set_title(
    'SHAP Feature Importance\n(XGBoost)', 
    fontsize=16, 
    pad=20, 
    loc='center'
)

# X-axis label 
ax.set_xlabel(
    "SHAP value (impact on model output)", 
    fontsize=14, 
    labelpad=10
)

# Adjust x-axis ticks
ax.tick_params(axis='x', labelsize=12)

# Fix colorbar aspect ratio
cbar_ax = plt.gcf().axes[-1]  
cbar_ax.set_aspect(20)        

# Adjust layout 
plt.tight_layout()
plt.subplots_adjust(
    bottom=0.2,  
    right=0.85   
)

# Show the plot
plt.show()
plt.savefig('working_path/figure_name.png', bbox_inches='tight')
plt.savefig('working_path/figure_name.pdf', bbox_inches='tight')
plt.close()