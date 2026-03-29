# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("loan_prediction.csv")

# Show first 5 rows
print(data.head())

# Show dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())


# Fill missing values
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mean(), inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)


# Convert categorical values to numbers
label = LabelEncoder()

data['Gender'] = label.fit_transform(data['Gender'])
data['Married'] = label.fit_transform(data['Married'])
data['Dependents'] = label.fit_transform(data['Dependents'])
data['Education'] = label.fit_transform(data['Education'])
data['Self_Employed'] = label.fit_transform(data['Self_Employed'])
data['Property_Area'] = label.fit_transform(data['Property_Area'])
data['Loan_Status'] = label.fit_transform(data['Loan_Status'])


# Drop Loan_ID column
data = data.drop('Loan_ID', axis=1)


# Split data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


sns.countplot(x='Loan_Status', data=data)
plt.show()


# ---------- User Prediction System ----------

print("\nEnter Applicant Details For Loan Prediction\n")

gender = int(input("Gender (Male=1 Female=0): "))
married = int(input("Married (Yes=1 No=0): "))
dependents = int(input("Dependents (0/1/2/3): "))
education = int(input("Education (Graduate=0 Not Graduate=1): "))
self_employed = int(input("Self Employed (Yes=1 No=0): "))
applicant_income = int(input("Applicant Income: "))
coapplicant_income = int(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Amount Term: "))
credit_history = int(input("Credit History (1=Good 0=Bad): "))
property_area = int(input("Property Area (Urban=2 Semiurban=1 Rural=0): "))

# Create input array
user_data = [[
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
]]

# Prediction
prediction = model.predict(user_data)

# Result
if prediction[0] == 1:
    print("\nLoan Approved ✅")
else:
    print("\nLoan Not Approved ❌")