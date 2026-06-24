import os

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet

import joblib
from joblib import Parallel, delayed
import time

class EN:
    def __init__(self):
    # Define models and parameter grids
        self.models_dict = {
            'ElasticNet': (
                ElasticNet(random_state=42),
                {
                    'alpha': np.logspace(-3, 1, 10),
                    'l1_ratio': np.linspace(0.1, 0.9, 9)
                }
            )}

    def EN_tuning(self, pipeline, Features_train, target_train, Features_test, target_test, 
                    models_dict, Inner_cv, i, eid_test, save_path=None, pipeline_save_path=None, n_jobs=-1):
        """
        Hyperparameter tuning
        Build ML pipelines

        X: Features
        y: target
        models_dict: dictionary of models and hyperparameters for tuning
        """
        start = time.time()
        
        # Initialize results dictionary (same as before)
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

            # Rebuild the pipeline 
            current_pipeline = Pipeline([
                ('preprocessor', pipeline.named_steps['preprocessor']),
                ('deconfound', pipeline.named_steps['deconfound']),
                ('model', model)  # update new model
            ])
            
            # Update parameter grid to include the model step
            scoped_param_grid = {f'model__{key}': value for key, value in param_grid.items()}

            # Grid search for hyperparameters
            grid_search = GridSearchCV(
                estimator=current_pipeline,
                param_grid=scoped_param_grid,
                cv=Inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0, 
                error_score='raise'
            )

            # Fit on training data
            grid_search.fit(Features_train, target_train)

            # Get model performance for training set to check potential overfitting
            y_pred_train = grid_search.predict(Features_train)
            r2_train = r2_score(target_train, y_pred_train)
            mse_train = mean_squared_error(target_train, y_pred_train)
            mae_train = mean_absolute_error(target_train, y_pred_train)
            rmse_train = root_mean_squared_error(target_train, y_pred_train)
            r_pear_train = np.corrcoef(target_train, y_pred_train)[0, 1]
            spearman_train = spearmanr(target_train, y_pred_train)
            r_spear_train = spearman_train.correlation

            # Get best model
            best_model = grid_search.best_estimator_

            # Save the full pipeline
            if pipeline_save_path:  # Only save if directory is provided
                os.makedirs(pipeline_save_path, exist_ok=True)
                full_pipeline_path = os.path.join(
                    pipeline_save_path,
                    f"{model_name}_fold_{i+1}_pipeline.joblib"
                )
                joblib.dump(best_model, full_pipeline_path)
                print(f"Saved pipeline to: {full_pipeline_path}")

            # Evaluate on outer test fold
            y_pred = best_model.predict(Features_test)

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
            results['best_params'].append(grid_search.best_params_)

            # Create prediction data for this model
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
            
            # Append metrics if file exists, otherwise create new
            if os.path.exists(save_path):
                try:
                    # Check if file is empty 
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

            # Save the predictions to a separate file
            base, ext = os.path.splitext(save_path)
            predictions_path = f"{base}_predictions{ext}"
            
            if os.path.exists(predictions_path):
                try:
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
