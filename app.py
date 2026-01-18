from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load Model and Preprocessor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inputs = {}

    if request.method == 'POST':
        try:
            inputs = {
                'Hours Studied': float(request.form.get('hours_studied', 0)),
                'Previous Scores': float(request.form.get('previous_scores', 0)),
                'Extracurricular Activities': request.form.get('extracurricular_activities'),
                'Sleep Hours': float(request.form.get('sleep_hours', 0)),
                'Sample Question Papers Practiced': float(request.form.get('sample_papers', 0))
            }

            input_df = pd.DataFrame([inputs])
            input_processed = preprocessor.transform(input_df)
            pred_value = model.predict(input_processed)[0]

            prediction = f"{pred_value:.2f}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, inputs=inputs)
