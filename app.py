from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("model/model_random_forest.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['GarageCars']),
            float(request.form['TotalBsmtSF']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]

        # Convert to 2D array for prediction
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
