
#!/usr/bin/env python3
"""
Preprocess for UKB variables.
We revised the direction of the variables to make sure that the higher score indicates a worse outcome for explaination.
We changed the name of chronotype and getting up in morning for clearness and consistency with the scores.
"""

import os
import numpy as np
import pandas as pd

# Load cross-sectional data (without any longitudinal data at instance 0)
df_cross = pd.read_csv('working_path/crosssectional.csv')
# Load clinical data 
df_clinical = pd.read_csv('working_path/clinical.csv')
# Load longitudinal data (with longitudinal data at instance 0 and 2)
df_long = pd.read_csv('working_path/longitudinal.csv')

#%%
# Set visit for variables in cross-sectional and clinical study
visit = '0.0'

# Define variables
Depression = ['depression_variables']
Sleep = ['sleep_variables']
Lifestyle_alco = ['alcohol_variables']
Lifestyle_smok = ['smoke_variables']
Lifestyle_diet = ['diet_variables']
Lifestyle_acti = ['activity_variables'] 
Lifestyle_electro =['electronic_variables']
Lifestyle_sun = ['sun_variables']
Employment_status = ['employment_variables']
Social_support = ['social_support_variables']

checklist_var = Depression + Sleep + Lifestyle_acti + Lifestyle_smok + Lifestyle_alco + Lifestyle_diet + Lifestyle_electro + Lifestyle_sun + Employment_status + Social_support

# Define the variables need to be reversed.
Reversed_sleep = [f'Getting_up_in_morning-{visit}'] # the name should be changed to Difficult_to_get_up_in_morning
Reversed_Lifestyle_alco = [f'Alcohol_intake_frequency.-{visit}']
Reversed_snoring = [f'Snoring-{visit}']
Reversed_social_support = [f'Frequency_of_friend/family_visits-{visit}']

# Define the list to be reversed
Reverse_var = Reversed_sleep + Reversed_Lifestyle_alco + Reversed_snoring + Reversed_social_support 
#%%
# Convert variables with likert scales to the same trend in cross-sectional dataframe
df_copy = df_cross.copy()
for column in df_copy[Reverse_var]:
    max_value = df_copy[column].max()
    if  max_value > 6:  # Frequency of friend/family visits
        min_value, max_value = 1, 7
    elif 4 < max_value <= 6: # Alcohol intake frequency
        min_value, max_value = 1, 6
    elif 2 < max_value <= 4: # Getting up in morning
        min_value, max_value = 1, 4
    else:
        min_value, max_value = 1, 2 # snoring
    df_copy[column] =  min_value + max_value - df_copy[column]
reversed_df_cross = df_copy

#%%
# To check if the scale trend in checklist_var has been reversed
for column in df_cross[checklist_var]:
    print('Before the reversion in cross-sectional dataframe:\n')
    print({column})
    print(df_cross[column].head())
    print('After the reversion in cross-sectional dataframe:\n')
    print(reversed_df_cross[column].head())
    print("================================================================================================================================")
# %%
# Convert variables with likert scales to the same trend in clinical dataframe
df_copy = df_clinical.copy()
for column in df_copy[Reverse_var]:
    max_value = df_copy[column].max()
    if  max_value > 6:  # Frequency of friend/family visits
        min_value, max_value = 1, 7
    elif 4 < max_value <= 6: # Alcohol intake frequency
        min_value, max_value = 1, 6
    elif 2 < max_value <= 4: # Getting up in morning
        min_value, max_value = 1, 4
    else:
        min_value, max_value = 1, 2 # snoring
    df_copy[column] =  min_value + max_value - df_copy[column]
reversed_df_clinical = df_copy

#%%
# To check if the scale trend in checklist_var has been reversed
for column in df_clinical[checklist_var]:
    print('Before the reversion in clinical dataframe:\n')
    print({column})
    print(df_clinical[column].head())
    print('After the reversion in clinical dataframe:\n')
    print(reversed_df_clinical[column].head())
    print("================================================================================================================================")

# %%
# Convert variables with likert scales to the same trend in longitudinal dataframe
df_copy = df_long.copy()
for column in df_copy[Reverse_var]:
    max_value = df_copy[column].max()
    if  max_value > 6:  # Frequency of friend/family visits
        min_value, max_value = 1, 7
    elif 4 < max_value <= 6: # Alcohol intake frequency
        min_value, max_value = 1, 6
    elif 2 < max_value <= 4: # Getting up in morning
        min_value, max_value = 1, 4
    else:
        min_value, max_value = 1, 2 # snoring
    df_copy[column] =  min_value + max_value - df_copy[column]
reversed_df_longitudinal = df_copy

#%%
# To check if the scale trend in checklist_var has been reversed
for column in df_long[checklist_var]:
    print('Before the reversion in cross-sectional dataframe:\n')
    print({column})
    print(df_long[column].head())
    print('After the reversion in cross-sectional dataframe:\n')
    print(reversed_df_longitudinal[column].head())
    print("================================================================================================================================")

#%%
# Change the name of chronotype and getting up in morning in cross-sectional data for clearness and consistency with the scores
# Create a mapping dictionary for renaming
rename_mapping = {
    f'Getting_up_in_morning-{visit}': f'Difficult_to_get_up_in_morning-{visit}',
    f'Morning/evening_person_(chronotype)-{visit}': f'Evening_chronotype-{visit}'
}

# Update the sleep list with new names
Sleep = [rename_mapping.get(item, item) for item in Sleep]

# Rename the columns in the Reversed data
reversed_df_cross = reversed_df_cross.rename(columns = rename_mapping)

# Check the changed names in sleep
print(reversed_df_cross[Sleep].head())

# Change the name of chronotype and getting up in morning in clinical dataframe
# Rename the columns in the Reversed dataframe
reversed_df_clinical = reversed_df_clinical.rename(columns = rename_mapping)

# Check the changed names in sleep
print(reversed_df_clinical[Sleep].head())

# Rename the columns in the Reversed dataframe
reversed_df_longitudinal = reversed_df_longitudinal.rename(columns = rename_mapping)

# Check the changed names in sleep
print(reversed_df_longitudinal[Sleep].head())

# %%
'''
Convert the employment status to working and non-working in cross-sectional data
1	In paid employment or self-employed
2	Retired
3	Looking after home and/or family
4	Unable to work because of sickness or disability
5	Unemployed
6	Doing unpaid or voluntary work
7	Full or part-time student
'''
# Convert employment status variables score either 1, 6, or 7 to working, otherwise, non-working
Working_list = [1, 6, 7]
Employment_vars_all = [col for col in reversed_df_cross.columns if col.startswith('Current_employment_status-0.')]

# Create binary employment_category (1=working, 0=non-working)
reversed_df_cross['Working_status'] = 0  # Default: non-working
reversed_df_cross.loc[reversed_df_cross[Employment_vars_all].isin(Working_list).any(axis=1), 'Working_status'] = 1
print(f"/nworking status in cross-sectional data: {reversed_df_cross['Working_status'].value_counts()}")
print(f"Missing values in working status in cross-sectional data: {reversed_df_cross['Working_status'].isna().any()}")
# %%
'''
Convert the employment status to working and non-working in clinical data
'''
# Convert employment status variables score either 1, 6, or 7 to working, otherwise, non-working
Working_list = [1, 6, 7]
Employment_vars_all = [col for col in reversed_df_clinical.columns if col.startswith('Current_employment_status-0.')]

# Create binary employment_category (1=working, 0=non-working)
reversed_df_clinical['Working_status'] = 0  # Default: non-working
reversed_df_clinical.loc[reversed_df_clinical[Employment_vars_all].isin(Working_list).any(axis=1), 'Working_status'] = 1
print(f"working status in clinical data: {reversed_df_clinical['Working_status'].value_counts()}")
print(f"Missing values in working status in clinical data: {reversed_df_clinical['Working_status'].isna().any()}")

'''
Convert the employment status to working and non-working in longitudinal data
'''
# Convert employment status variables score either 1, 6, or 7 to working, otherwise, non-working
Working_list = [1, 6, 7]
Employment_vars_all = [col for col in reversed_df_longitudinal.columns if col.startswith('Current_employment_status-0.')]

# Create binary employment_category (1=working, 0=non-working)
reversed_df_longitudinal['Working_status'] = 0  # Default: non-working
reversed_df_longitudinal.loc[reversed_df_longitudinal[Employment_vars_all].isin(Working_list).any(axis=1), 'Working_status'] = 1
print(f"working status in longitudinal data: {reversed_df_longitudinal['Working_status'].value_counts()}")
print(f"Missing values in working status in longitudinal data: {reversed_df_longitudinal['Working_status'].isna().any()}")

###########################################################################################
#==========================================================================================
###########################################################################################

# Save the reversed dataframes to CSV files
reversed_df_cross.to_csv('working_path/cross_reversed.csv')
reversed_df_clinical.to_csv('working_path/clinical_reversed.csv')
reversed_df_longitudinal.to_csv('working_path/long_reversed.csv')
