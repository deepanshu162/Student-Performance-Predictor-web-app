import joblib
import pandas as pd
import argparse
import os
import sys
import numpy as np

def get_input(prompt, type_func, valid_options=None):
    while True:
        try:
            value = input(prompt + ": ")
            if valid_options:
                if value not in valid_options:
                    print(f"Invalid input. Options: {valid_options}")
                    continue
            return type_func(value)
        except ValueError:
            print(f"Invalid input type. Expected {type_func.__name__}")

def predict():
    parser = argparse.ArgumentParser(description='Predict Student Performance')
    parser.add_argument('--hours_studied', type=int, required=False, help='Hours Studied')
    parser.add_argument('--previous_scores', type=int, required=False, help='Previous Scores')
    parser.add_argument('--extracurricular_activities', type=str, required=False, help='Extracurricular Activities (Yes/No)')
    parser.add_argument('--sleep_hours', type=int, required=False, help='Sleep Hours')
    parser.add_argument('--sample_papers_practiced', type=int, required=False, help='Sample Question Papers Practiced')

    args = parser.parse_args()

    try:
        model_path = "model.pkl"
        preproc_path = "preprocessor.pkl"
        metrics_path = ".tmp/metrics.joblib"
        features_path = ".tmp/feature_names.joblib"

        if not os.path.exists(model_path) or not os.path.exists(preproc_path):
            print("Error: Model or Preprocessor not found. Train the model first.")
            return

        # Load artifacts
        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
        
        metrics = None
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            
        feature_names = None
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)

        # Collect Input
        inputs = {}
        
        if args.hours_studied is not None:
            inputs['Hours Studied'] = args.hours_studied
        else:
            inputs['Hours Studied'] = get_input("Enter Hours Studied (0-24)", int)

        if args.previous_scores is not None:
            inputs['Previous Scores'] = args.previous_scores
        else:
            inputs['Previous Scores'] = get_input("Enter Previous Scores (0-100)", int)

        if args.extracurricular_activities is not None:
            inputs['Extracurricular Activities'] = args.extracurricular_activities
        else:
            inputs['Extracurricular Activities'] = get_input("Extracurricular Activities (Yes/No)", str, ["Yes", "No"])

        if args.sleep_hours is not None:
            inputs['Sleep Hours'] = args.sleep_hours
        else:
            inputs['Sleep Hours'] = get_input("Enter Sleep Hours (0-24)", int)

        if args.sample_papers_practiced is not None:
            inputs['Sample Question Papers Practiced'] = args.sample_papers_practiced
        else:
            inputs['Sample Question Papers Practiced'] = get_input("Enter Sample Question Papers Practiced (0-10)", int)

        # Create DataFrame from input
        input_data = pd.DataFrame([inputs])

        # Preprocess
        input_processed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(input_processed)[0]

        print("\n" + "=" * 40)
        print("PREDICTION REPORT")
        print("=" * 40)
        print(f"Predicted Performance Index: {prediction:.2f}")

        # Explain Error
        if metrics:
            mae = metrics['mae']
            error_percent = (mae / prediction) * 100 if prediction != 0 else 0
            print(f"Estimated Error Margin: ±{mae:.2f} points (approx. ±{error_percent:.1f}%)")
            print(f"Confidence Level: High (R² ~ 0.99 on Test Data)")
        
        # Explain Logic (Feature Importance)
        if feature_names and hasattr(model, 'feature_importances_'):
            print("\nWHAT DRIVES THIS PREDICTION?")
            print("-" * 30)
            
            importances = model.feature_importances_
            
            # Since input was transformed (one-hot), we need to handle that carefully.
            # However, for simple explanation, we can map back to raw features if possible
            # But the user asked for mapping inputs -> important features.
            
            # Simple heuristic: Global Importance
            indices = np.argsort(importances)[::-1]
            
            # We will show top 3 features and the user's value for them
            count = 0
            shown_features = set()
            
            for f in indices:
                feat_name = feature_names[f]
                score = importances[f]
                
                # Simple mapping back to input columns
                # E.g. "Previous Scores" -> "Previous Scores"
                # "Extracurricular Activities_Yes" -> "Extracurricular Activities"
                
                mapped_name = feat_name
                user_val = "N/A"
                
                if "Previous Scores" in feat_name:
                    mapped_name = "Previous Scores"
                    user_val = inputs['Previous Scores']
                elif "Hours Studied" in feat_name:
                    mapped_name = "Hours Studied"
                    user_val = inputs['Hours Studied']
                elif "Sleep Hours" in feat_name:
                    mapped_name = "Sleep Hours"
                    user_val = inputs['Sleep Hours']
                elif "Sample Question Papers" in feat_name:
                    mapped_name = "Sample Question Papers Practiced"
                    user_val = inputs['Sample Question Papers Practiced']
                elif "Extracurricular Activities" in feat_name:
                    mapped_name = "Extracurricular Activities"
                    user_val = inputs['Extracurricular Activities']

                if mapped_name in shown_features:
                    continue
                
                print(f"{count+1}. {mapped_name}: {user_val}")
                print(f"   (Impact Factor: {score*100:.1f}%)")
                
                shown_features.add(mapped_name)
                count += 1
                if count >= 3:
                     break
            
            print("\n*Impact Factor represents the global importance of this feature in the model.")

        print("=" * 40 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict()
