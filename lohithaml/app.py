from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        input_data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        # Convert input data to a numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_array)
        
        # Predict the result using the model
        prediction = classifier.predict(std_data)
        
        # Show the result
        if prediction[0] == 1:
            return render_template('result.html', prediction="The person is diabetic.")
        else:
            return render_template('result.html', prediction="The person is not diabetic.")

if __name__ == '__main__':
    app.run(debug=True)