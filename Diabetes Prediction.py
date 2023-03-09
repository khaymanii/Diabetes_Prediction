
# Importing the Libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



# Data Collection and Analysis

diabetes_dataset = pd.read_csv('diabetes.csv')

diabetes_dataset.head()

diabetes_dataset.shape
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()


# Seperating Data and Labels

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
y = diabetes_dataset['Outcome']

print(X)
print(y)


# Data Standardization

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)


X = standardized_data
y = diabetes_dataset['Outcome']

print(X)
print(y)


# Train Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)


print(X.shape, X_train.shape, X_test.shape)

#Training the Model

classifier = svm.SVC(kernel = 'linear')

#Training the SVM classifier

classifier.fit(X_train, y_train)


# Model Evaluation : accuracy score of training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('accuracy score of training data ', training_data_accuracy)

# Model Evaluation : accuracy score of test data

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)


print('accuracy score of test data ', test_data_accuracy)


# Making a Predictive System

input_data = (0,137,40,35,168,43.1,2.288,33)

# Convert the input data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data

std_data = scaler.transform(input_data_reshaped)

print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

