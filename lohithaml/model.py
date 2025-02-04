import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
diabetes_dataset = pd.read_csv(r'C:\Users\lohit\OneDrive\Desktop\lohithaml\diabetes (3) (1).csv')

# Display dataset information
print("Dataset Shape:", diabetes_dataset.shape)
print("Null Values Check:\n", diabetes_dataset.isnull().sum())
print("Class Distribution:\n", diabetes_dataset['Outcome'].value_counts())

# Split the features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the feature data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)  # Corrected: fit and transform the data

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_standardized, Y, test_size=0.2, stratify=Y, random_state=2)

print(f"Dataset shapes - Full: {X.shape}, Training: {X_train.shape}, Testing: {X_test.shape}")

# Create and train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluate the model on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data:', training_data_accuracy)

# Evaluate the model on testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data:', test_data_accuracy)

# Save the model and scaler using pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")

# Define feature names
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Load the model and scaler once
with open('svm_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function for prediction based on input data
def predict_diabetes(input_data):
    # Convert input data to DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Standardize the input data
    std_data = scaler.transform(input_df)
    
    # Predict the result using the model
    prediction = classifier.predict(std_data)

    if prediction[0] == 1:
        print('The person is diabetic')
    else:
        print('The person is not diabetic')

# Test the model with sample input
input_data = (2, 85, 66, 29, 0, 26.6, 0.351, 31)  # Example input data
predict_diabetes(input_data)
