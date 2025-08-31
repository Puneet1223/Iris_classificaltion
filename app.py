from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load saved model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)
# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Convert into array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict using model
        prediction = model.predict(features)[0]

        return render_template("index.html",
                               prediction_text=f"Predicted Iris Species: {prediction}")
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
