"""
Custom Preprocessing & Feature Engineering Module for Mobile Price Classification.

Dataset: Mobile Price Classification
Source: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
Kaggle Dataset ID: iabhishekofficial/mobile-price-classification
Instances: 2000 | Features: 20 | Target: price_range (0, 1, 2, 3)

Author: Ajay Srivastava
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

SEED = 1307

class MobileDataPreprocessor:
    def __init__(self, data_path: str = "mobile_data.csv", random_state: int = SEED):
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_and_split_data(self, test_size: float = 0.2):
        """Loads mobile data and performs stratified split."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['price_range'])
        y = df['price_range']
        
        self.feature_columns = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def fit_transform_features(self, X_train, X_test, save_scaler_path: str = "model/mobile_price_scaler.joblib"):
        """Applies Z-score standard scaling and saves the fitted scaler."""
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        joblib.dump(self.scaler, save_scaler_path)
        return X_train_scaled, X_test_scaled

    def transform_single_instance(self, input_dict: dict, scaler_path: str = "model/mobile_price_scaler.joblib"):
        """Transforms a single feature dictionary for live Streamlit inference."""
        scaler = joblib.load(scaler_path)
        df_inst = pd.DataFrame([input_dict])
        return scaler.transform(df_inst)
