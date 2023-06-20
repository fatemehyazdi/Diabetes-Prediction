# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:11:34 2022

@author: Fatemeh
"""

#Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#______________________________________________________________________________
#Data Collection and Analysis
diabetes_dataset = pd.read_csv('diabetes.csv') 
print("diabetes_dataset :","\n", diabetes_dataset,"\n")

# printing the first 5 rows of the dataset
head = diabetes_dataset.head()
print("head :","\n", head,"\n")

# number of rows and Columns in this dataset
shape = diabetes_dataset.shape
print("shape :","\n", shape,"\n")

# getting the statistical measures of the data
#std --> standard deviation
#25% --> percentage of each Columns less than specific number in table
describe = diabetes_dataset.describe()
print("describe :","\n", describe,"\n")

#value_counts
#0 --> Non-Diabetic , 1 --> Diabetic
count = diabetes_dataset['Outcome'].value_counts()
print("count :","\n", count,"\n")


#mean value counts for each Columns
mean_value_counts = diabetes_dataset.groupby('Outcome').mean()
print("mean_value_counts :","\n", mean_value_counts,"\n")

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print("X :","\n", X,"\n")
print("Y :","\n", Y,"\n")

#______________________________________________________________________________
#Data preprocessing
#Data Standardization
#All these values in the range of zero and one(similar range)
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print("standardized_data :","\n", standardized_data,"\n")

# Data after Standardization
X = standardized_data
Y = diabetes_dataset['Outcome']
print("X :","\n", X,"\n")
print("Y :","\n", Y,"\n")

#______________________________________________________________________________
#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, 
                                                    stratify=Y, random_state=2)
print("X_train :","\n", X_train,"\n","X_test :","\n", X_test,"\n")

print("X.shape :","\n", X.shape,"\n",
      "X_train.shape :","\n", X_train.shape,"\n",
      "X_test.shape :","\n", X_test.shape,"\n")

#______________________________________________________________________________
#Training the Model
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#______________________________________________________________________________
#Model Evaluation
#accuracy score on the train data
X_train_prediction = classifier.predict(X_train)
print("X_train_prediction :","\n", X_train_prediction,"\n")

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data :","\n", training_data_accuracy,"\n")

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
print("X_test_prediction :","\n", X_test_prediction,"\n")

test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data :","\n", test_data_accuracy,"\n")

#______________________________________________________________________________
#Making a Predictive System
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
print("input_data_as_numpy_array :","\n", input_data_as_numpy_array,"\n")

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print("input_data_reshaped :","\n", input_data_reshaped,"\n")

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print("std_data :","\n", std_data,"\n")

prediction = classifier.predict(std_data)
print("prediction :","\n", prediction,"\n")

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
