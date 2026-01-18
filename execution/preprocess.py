import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def preprocess():
    file_path = 'StudentPerformance.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    try:
        df = pd.read_csv(file_path)
        
        target = 'Performance Index'
        if target not in df.columns:
             print(f"Error: Target '{target}' not found.")
             return
             
        X = df.drop(columns=[target])
        y = df[target]
        
        # Identify columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        # Create Preprocessing Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier debug/df conversion if needed
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit and Transform
        print("Fitting preprocessor...")
        X_processed = preprocessor.fit_transform(X)
        
        # Save artifacts
        if not os.path.exists(".tmp"):
            os.makedirs(".tmp")
            
        print("Saving artifacts...")
        joblib.dump(preprocessor, 'preprocessor.pkl')
        joblib.dump(X_processed, '.tmp/X_processed.joblib')
        joblib.dump(y, '.tmp/y.joblib')
        
        # Save feature names for later use in importance
        try:
             # This might fail depending on sklearn version/onehot config, but good to try
            feature_names = numeric_features + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
            joblib.dump(feature_names, '.tmp/feature_names.joblib')
        except:
            print("Could not extract feature names.")

        print("Preprocessing Complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preprocess()
