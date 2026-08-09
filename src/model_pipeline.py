"""
Hyperparameter Optimization & Training Pipeline.
Fine-tunes 5 classification models using 5-Fold Cross Validation.
Author: Ajay Srivastava
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import MobileDataPreprocessor, SEED
from src.evaluator import evaluate_classification_performance

def run_training_and_tuning_pipeline():
    print(f"Initializing Mobile Data Preprocessor with random seed = {SEED}...")
    preprocessor = MobileDataPreprocessor(data_path="mobile_data.csv", random_state=SEED)
    X_train, X_test, y_train, y_test = preprocessor.load_and_split_data(test_size=0.2)
    
    # Save test data files for Streamlit deployment verification
    test_df_with_labels = pd.concat([X_test, y_test], axis=1)
    test_df_with_labels.to_csv("test_data.csv", index=False)
    X_test.to_csv("test_data_features.csv", index=False)
    print("Saved test_data.csv and test_data_features.csv.")
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.fit_transform_features(
        X_train, X_test, save_scaler_path="model/mobile_price_scaler.joblib"
    )
    
    # Define custom hyperparameter search grids for each model
    model_configurations = {
        'Logistic Regression': {
            'estimator': LogisticRegression(max_iter=1500, random_state=SEED),
            'param_grid': {
                'C': [0.05, 0.5, 2.0, 15.0, 50.0, 100.0],
                'solver': ['lbfgs', 'saga'],
                'multi_class': ['multinomial', 'ovr']
            },
            'save_filename': 'model/logistic_regression_model.joblib'
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=SEED),
            'param_grid': {
                'max_depth': [4, 6, 9, 14, None],
                'min_samples_split': [2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'save_filename': 'model/decision_tree_model.joblib'
        },
        'kNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [5, 9, 13, 17, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'save_filename': 'model/knn_classifier_model.joblib'
        },
        'Naive Bayes': {
            'estimator': GaussianNB(),
            'param_grid': {
                'var_smoothing': np.logspace(-1, -9, num=50)
            },
            'save_filename': 'model/naive_bayes_model.joblib'
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=SEED),
            'param_grid': {
                'n_estimators': [80, 120, 200],
                'max_depth': [6, 12, None],
                'min_samples_split': [2, 6],
                'criterion': ['gini', 'entropy']
            },
            'save_filename': 'model/random_forest_model.joblib'
        }
    }
    
    benchmark_results = []
    
    print("\n--- Starting 5-Fold Cross Validation Hyperparameter Tuning ---")
    for model_name, config in model_configurations.items():
        print(f"\nTuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Optimal Hyperparameters: {grid_search.best_params_}")
        
        # Save model artifact
        joblib.dump(best_model, config['save_filename'])
        print(f"Artifact saved: {config['save_filename']}")
        
        # Predict on holdout test set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled) if hasattr(best_model, "predict_proba") else None
        
        # Evaluate metrics
        metrics = evaluate_classification_performance(model_name, y_test, y_pred, y_prob)
        benchmark_results.append(metrics)
        print(f"Test Accuracy: {metrics['Accuracy']} | AUC: {metrics['AUC']} | F1: {metrics['F1']} | MCC: {metrics['MCC']}")
        
    df_metrics = pd.DataFrame(benchmark_results)
    print("\n================ FINAL EVALUATION METRICS TABLE ================")
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv("model/model_metrics.csv", index=False)
    return df_metrics

if __name__ == '__main__':
    run_training_and_tuning_pipeline()
