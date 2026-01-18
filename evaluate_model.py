import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model():
    try:
        if not os.path.exists("model.pkl"):
             print("Error: Model not found. Run train_model.py first.")
             return
        if not os.path.exists(".tmp/X_test.joblib") or not os.path.exists(".tmp/y_test.joblib"):
             print("Error: Test data not found. Run train_model.py first.")
             return

        print("Loading model and data...")
        model = joblib.load('model.pkl')
        X_test = joblib.load('.tmp/X_test.joblib')
        y_test = joblib.load('.tmp/y_test.joblib')
        X_train = joblib.load('.tmp/X_train.joblib')
        y_train = joblib.load('.tmp/y_train.joblib')

        # Predict
        print("Predicting...")
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Metrics
        def calculate_metrics(y_true, y_pred, name):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"\n{name} Metrics:")
            print(f"R2 Score: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            return r2

        r2_train = calculate_metrics(y_train, y_pred_train, "Train")
        r2_test = calculate_metrics(y_test, y_pred_test, "Test")

        print("-" * 20)
        print(f"Overfitting Check (Train R2 - Test R2): {r2_train - r2_test:.4f}")

        # Save Metrics for Predictor
        metrics = {
             "mae": mean_absolute_error(y_test, y_pred_test),
             "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        joblib.dump(metrics, '.tmp/metrics.joblib')
        print("Metrics saved to .tmp/metrics.joblib")
        
        # Feature Importance
        print("\nFeature Importance:")
        try:
            if os.path.exists(".tmp/feature_names.joblib"):
                feature_names = joblib.load(".tmp/feature_names.joblib")
                importances = model.feature_importances_
                
                # Sort and ensure lengths match
                if len(feature_names) == len(importances):
                    indices = np.argsort(importances)[::-1]
                    for f in range(len(feature_names)):
                        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
                else:
                    print("Feature names length mistmatch.")
            else:
                 print("Feature names file not found.")
        except Exception as fi_e:
            print(f"Could not print feature importance: {fi_e}")

        print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()
