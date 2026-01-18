import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model():
    try:
        if not os.path.exists(".tmp/X_processed.joblib") or not os.path.exists(".tmp/y.joblib"):
             print("Error: Preprocessed data not found. Run preprocess.py first.")
             return

        print("Loading preprocessed data...")
        X = joblib.load('.tmp/X_processed.joblib')
        y = joblib.load('.tmp/y.joblib')

        # Train Test Split
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        print("Training Random Forest Regressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save Model
        print("Saving model...")
        joblib.dump(model, 'model.pkl', compress=3)
        
        # Save Reference for Evaluation
        joblib.dump(X_train, '.tmp/X_train.joblib')
        joblib.dump(y_train, '.tmp/y_train.joblib')
        joblib.dump(X_test, '.tmp/X_test.joblib')
        joblib.dump(y_test, '.tmp/y_test.joblib')
        
        print("Training Complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()
