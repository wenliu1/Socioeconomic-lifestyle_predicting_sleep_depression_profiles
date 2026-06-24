import os

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import xgboost as xgb
import optuna

import joblib
from joblib import Parallel, delayed

import time

class XGBoptuna:
    def __init__(self, n_trials=150):
    # Define models and their parameter grids
        self.models_dict = {
            'XGBoost': (
            xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            {
            'n_estimators': (300, 1500),
            'learning_rate': (0.005, 0.1, 'log-uniform'),
            'max_depth': (2, 10),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1), 
            'gamma': (0, 1), 
            'reg_alpha': (0, 10), 
            'reg_lambda': (0, 10)
            }
        )}
        self.n_trials = n_trials

    def objective(self, trial, pipeline, Features_train, target_train, Features_test, target_test, model_name, model):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1), 
            'gamma': trial.suggest_float('gamma', 0, 1), 
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10), 
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10), 
            'random_state': 42,
            'objective': 'reg:squarederror'
        }

        # Inner fold for model performance for eatly pruning
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        mse_scores = []

        # Evaluate model performance in the inner fold
        for step, (train_idx, val_idx) in enumerate(kf.split(Features_train)):
            X_train = Features_train.iloc[train_idx] if isinstance(Features_train, pd.DataFrame) else Features_train[train_idx]
            X_val = Features_train.iloc[val_idx] if isinstance(Features_train, pd.DataFrame) else Features_train[val_idx]
            y_train = target_train.iloc[train_idx] if isinstance(target_train, pd.Series) else target_train[train_idx]
            y_val = target_train.iloc[val_idx] if isinstance(target_train, pd.Series) else target_train[val_idx]

            # Fresh pipeline 
            current_model = xgb.XGBRegressor(**params) 

            # Update the pipeline with the new model
            current_pipeline = Pipeline([
                ('preprocessor', pipeline.named_steps['preprocessor']),
                ('deconfound', pipeline.named_steps['deconfound']),
                ('model', current_model)  # Add new model
            ])

            # Fit on training data
            current_pipeline.fit(X_train, y_train)
            # Evaluate the model
            y_pred_val = current_pipeline.predict(X_val)
            # Loss measure on mse
            mse = mean_squared_error(y_val, y_pred_val)
            mse_scores.append(mse)

            # Report intermediate results from validation folds
            trial.report(mse, step=step)
            # Check for pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(mse_scores)

    def XGBoptuna_tuning(self, pipeline, Features_train, target_train, Features_test, target_test, 
                            models_dict, Inner_cv, i, eid_test, save_path=None, pipeline_save_path=None, optuna_n_jobs=1, n_jobs=1):
        """
        Hyperparameter tuning using optuna
        Build ML pipelines

        X: Features
        y: target
        models_dict: dictionary of models and hyperparameters for tuning
        """
        start = time.time()
        
        # Initialize results dictionary 
        results = {
            'model_name': [],
            'outer_fold': [],
            'train_r2': [],
            'train_mse': [],
            'train_mae': [],
            'train_rmse': [],
            'train_r_pear': [],
            'train_r_spear': [],
            'test_r2': [],
            'test_mse': [],
            'test_mae': [],
            'test_rmse': [],
            'test_r_pear': [],
            'test_r_spear': [],
            'best_params': []
        }
        
        # Create a separate list for y_prediction data
        results_list_prediction = []

        def evaluate_model(model_name, model_tuple):
            model, param_grid = model_tuple # Unpack the tuple
            print(f'\nEvaluating {model_name} on fold {i+1}:')

            # Create study object
            study = optuna.create_study(study_name=f'XGBoost_fold_{i+1}', 
                                        direction='minimize',
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, 
                                                                           n_warmup_steps=2,
                                                                           interval_steps=2))

            # Optimize the study 
            study.optimize(
                lambda trial: self.objective(
                    trial, pipeline, Features_train, target_train, Features_test, target_test, model_name, model
                ),
                n_trials=self.n_trials,
                n_jobs=optuna_n_jobs # Optuna-specific parallelism
            )
            
            # Get best parameters
            best_params = study.best_params
            print(f'Best parameters: {best_params}')

            # Train final model with best parameters
            final_model = model.set_params(**best_params)

            # Rebuild the pipeline with the best model
            final_pipeline = Pipeline([
                ('preprocessor', pipeline.named_steps['preprocessor']),
                ('deconfound', pipeline.named_steps['deconfound']),
                ('model', final_model)  
            ])
            
            # Fit on full training data
            final_pipeline.fit(Features_train, target_train)

            # Save the full pipeline
            if pipeline_save_path:  
                os.makedirs(pipeline_save_path, exist_ok=True)
                full_pipeline_path = os.path.join(
                    pipeline_save_path,
                    f"{model_name}_fold_{i+1}_pipeline_CV.joblib"
                )
                joblib.dump(final_pipeline, full_pipeline_path)
                print(f"Saved pipeline to: {full_pipeline_path}")

            # Get model performance for training set to check potential overfitting
            y_pred_train = final_pipeline.predict(Features_train)
            r2_train = r2_score(target_train, y_pred_train)
            mse_train = mean_squared_error(target_train, y_pred_train)
            mae_train = mean_absolute_error(target_train, y_pred_train)
            rmse_train = root_mean_squared_error(target_train, y_pred_train)
            r_pear_train = np.corrcoef(target_train, y_pred_train)[0, 1]
            spearman_train = spearmanr(target_train, y_pred_train)
            r_spear_train = spearman_train.correlation

            # Evaluate on outer test fold
            y_pred = final_pipeline.predict(Features_test)

            # Calculate evaluation metrics
            r2_test = r2_score(target_test, y_pred)
            mse_test = mean_squared_error(target_test, y_pred)
            mae_test = mean_absolute_error(target_test, y_pred)
            rmse_test = root_mean_squared_error(target_test, y_pred)
            r_pear_test = np.corrcoef(target_test, y_pred)[0, 1]
            spearman_result = spearmanr(target_test, y_pred)
            r_spear_test = spearman_result.correlation

            # Print out the results
            print(f"Training - R²: {r2_train:.4f}, r_pear: {r_pear_train:.4f}, r_spear: {r_spear_train:.4f}, MSE: {mse_train:.4f}, rMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
            print(f"Test - R²: {r2_test:.4f}, r_pear: {r_pear_test:.4f}, r_spear: {r_spear_test:.4f}, MSE: {mse_test:.4f}, rMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

            # Append results to the dictionary
            results['model_name'].append(model_name)
            results['outer_fold'].append(i+1)
            results['train_r2'].append(r2_train)
            results['train_mse'].append(mse_train)
            results['train_mae'].append(mae_train)
            results['train_rmse'].append(rmse_train)
            results['train_r_pear'].append(r_pear_train)
            results['train_r_spear'].append(r_spear_train)
            results['test_r2'].append(r2_test)
            results['test_mse'].append(mse_test)
            results['test_mae'].append(mae_test)
            results['test_rmse'].append(rmse_test)
            results['test_r_pear'].append(r_pear_test)
            results['test_r_spear'].append(r_spear_test)
            results['best_params'].append(best_params)

            # Create prediction data for the model
            for pred, true, eid_val in zip(y_pred, target_test, eid_test):
                prediction_data = {
                    'model_name': model_name,
                    'outer_fold': i+1,
                    'eid': eid_val,
                    'y_pred': pred,
                    'y_true': true
                }
                results_list_prediction.append(prediction_data)

        # Extract models and run evaluations (parallel or sequential)
        if n_jobs == 1: # Run sequentially
            for model_name, model_tuple in models_dict.items():
                evaluate_model(model_name, model_tuple)
        else: # Run in parallel
            Parallel(n_jobs=n_jobs)(
                delayed(evaluate_model)(model_name, model_tuple)
                for model_name, model_tuple in models_dict.items()
            )

        results_df = pd.DataFrame(results)
        print(f"Total time for all models: {time.time() - start:.2f} seconds")

        # Save the results
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Append the results if file exists, otherwise create new
            if os.path.exists(save_path):
                try:
                    if os.path.getsize(save_path) > 0:
                        existing_df = pd.read_csv(save_path)
                        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                    else:
                        combined_df = results_df
                except Exception as e:
                    print(f"Warning: Issue with reading existing file: {e}. Creating new file.")
                    combined_df = results_df
                combined_df.to_csv(save_path, index=False)
            else:
                results_df.to_csv(save_path, index=False)
            print(f"Metrics saved/appended to {save_path}")

            # Save the predictions to a separate CSV file
            base, ext = os.path.splitext(save_path)
            predictions_path = f"{base}_predictions{ext}"
            
            if os.path.exists(predictions_path):
                try:
                    # Check if file is empty 
                    if os.path.getsize(predictions_path) > 0:
                        existing_preds = pd.read_csv(predictions_path)
                        combined_preds = pd.concat([existing_preds, pd.DataFrame(results_list_prediction)], ignore_index=True)
                    else:
                        combined_preds = pd.DataFrame(results_list_prediction)
                except Exception as e:
                    print(f"Warning: Issue with reading existing file: {e}. Creating new file.")
                    combined_preds = pd.DataFrame(results_list_prediction)
                combined_preds.to_csv(predictions_path, index=False)
            else:
                pd.DataFrame(results_list_prediction).to_csv(predictions_path, index=False)
            print(f"Predictions saved/appended to {predictions_path}")

        return results_df
