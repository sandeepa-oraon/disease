from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import csv

import os

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(__file__)

# Specify the relative path to the CSV file
csv_file_path = os.path.join(script_dir, 'heart_disease_dataset_1.csv')

# Open the CSV file
with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    # Continue with your code...

# with open('predict\heart_disease_dataset_1.csv', 'r') as file:
#     reader = csv.reader(file)

df= pd.read_csv('heart_disease_dataset_1.csv')
#heartData = allData.sample(n = 5000, random_state = 42) #choosing 50000 random data points
minority_class = df[df['target'] == 0]
majority_class = df[df['target'] == 1]

# Check the number of instances in each class
n_minority = len(minority_class)
n_majority = len(majority_class)

# We need 2500 instances of each class
n_required = 2500

# Oversample the minority class using resample if the minority class is less than 2500
if n_minority < n_required:
    minority_class_oversampled = resample(minority_class,
                                          replace=True,    # Sample with replacement
                                          n_samples=n_required, # Number of samples to match the majority class
                                          random_state=42)  # Seed for reproducibility
else:
    minority_class_oversampled = minority_class.sample(n=n_required, random_state=42)

# Randomly sample the majority class
majority_class_sampled = majority_class.sample(n=n_required, random_state=42)

# Combine the oversampled minority class with the sampled majority class
balanced_df = pd.concat([minority_class_oversampled, majority_class_sampled])

# Shuffle the resulting dataset
heart_data = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# The resulting dataset balanced_df now contains 5000 instances with an equal number of 0s and 1s
# print(heart_data['target'].value_counts())
y= heart_data['target']
x= heart_data.drop('target', axis= 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

svm_model1= SVC()
svm_model1.fit(x_train, y_train)

y_train_pred_svc= svm_model1.predict(x_train)
#make predection on the test set
y_test_pred_svc = svm_model1.predict(x_test)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(svm_model1, file)

app = Flask(__name__)

# Load your machine learning model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    HighBP = int(request.form['HighBP'])
    HighChol = int(request.form['HighChol'])
    BMI = float(request.form['BMI'])
    Smoker = int(request.form['Smoker'])
    Stroke = int(request.form['Stroke'])
    PhysActivity = int(request.form['PhysActivity'])
    AnyHealthcare = int(request.form['AnyHealthcare'])
    GenHlth = int(request.form['GenHlth'])
    PhysHlth = int(request.form['PhysHlth'])
    DiffWalk = int(request.form['DiffWalk'])
    Sex = int(request.form['Sex'])
    Age = int(request.form['Age'])
    SleepTime = int(request.form['SleepTime'])
    SkinCancer = int(request.form['SkinCancer'])
    Asthma = int(request.form['Asthma'])
    KidneyDisease = int(request.form['KidneyDisease'])
    
    # Create input array
    input_data = np.array([HighBP, HighChol, BMI, Smoker, Stroke, PhysActivity, AnyHealthcare,
                           GenHlth, PhysHlth, DiffWalk, Sex, Age, SleepTime, SkinCancer, Asthma, KidneyDisease])
    input_reshaped = input_data.reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(input_reshaped)

    result = "The person does not have heart disease" if prediction[0] == 0 else "The person has heart disease"

    return render_template('disease.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
