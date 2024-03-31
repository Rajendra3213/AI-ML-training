import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import numpy as np

feature_columns = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL',
                   'CREA', 'GGT', 'PROT']
label = 'Category_numeric'

st.write("""
# Machine Learning 
Predicting Hepatitis C
""")

feature_values = {}

st.sidebar.header('Enter Feature Values')
for feature in feature_columns:
    feature_values[feature] = st.sidebar.number_input(
        f'Enter value for {feature}', step=0.01)

with open('model_final.pkl', 'rb') as f:
    model = joblib.load(f)

if st.sidebar.button('Submit'):
    st.write('Entered Feature Values:')
    st.write(feature_values)
    # Convert feature values to DataFrame
    feature_values_df = pd.DataFrame([feature_values])

    # Make predictions
    predictions = model.predict(feature_values_df)

    with open('label_encoder_final.pkl', 'rb') as f:
        label_encoder = joblib.load(f)

    # Ensure label_encoder is of type LabelEncoder
    # assert isinstance(label_encoder, LabelEncoder), "label_encoder is not an instance of LabelEncoder"
    predicted_label = label_encoder.inverse_transform(predictions)

    # st.write(predictions)
    st.write(predicted_label)


file_path = "HepatitisCdata 2.csv"
data = pd.read_csv(file_path, index_col=0)
st.write("Given Data")
st.write(data)

num_columns = len(data.columns)
num_rows = (num_columns + 2) // 3

st.write("Data Distribution")
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))
for i, column in enumerate(data.columns):
    row = i // 3
    col = i % 3
    axes[row, col].hist(data[column], bins=20)
    axes[row, col].set_title(column)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')

if num_columns % 3 != 0:
    for i in range(num_columns % 3, 3):
        fig.delaxes(axes[num_rows - 1, i])

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

plt.figure(figsize=(8, 6))
distribution = sns.countplot(data=data, x="Category")
plt.title("Diagnosis Distribution")
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[
           "Absent", "Suspected", "Stage 1", "Stage 2", "Stage 3"])
st.pyplot(plt)

data["Sex"] = data["Sex"].replace("m", 1)
data["Sex"] = data["Sex"].replace("f", 0)

label_encoder1 = LabelEncoder()
data["Category_numeric"] = label_encoder1.fit_transform(data["Category"])

data_subset = data.drop(columns=['Category'])

st.write("Cleaning Data: after clearing all null values and replacing them with mean value")
data_subset['ALB'].fillna(data_subset['ALB'].mean(), inplace=True)
data_subset['ALP'].fillna(data_subset['ALP'].mean(), inplace=True)
data_subset['CHOL'].fillna(data_subset['CHOL'].mean(), inplace=True)
data_subset['PROT'].fillna(data_subset['PROT'].mean(), inplace=True)
data_subset['ALT'].fillna(data_subset['ALT'].mean(), inplace=True)
st.write(data_subset.head())

st.write("Correlation")
corr_matrix = data_subset.corr()
st.write(corr_matrix)

st.write("Heat Map")
plt.figure(figsize=(15, 8))
table = sns.heatmap(data_subset.corr(), annot=True, fmt=".2f", cmap="coolwarm")
st.pyplot(plt)

X = data_subset[feature_columns]
y = data_subset[label]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

st.title("Training Data")
col1, col2 = st.columns(2)

with col1:
    st.write("Train Data Set")
    st.write(X_train)

with col2:
    st.write("Train Labels")
    st.write(y_train)

# # logistic_model = LogisticRegression()
# logistic_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
st.pyplot(plt)

# Store the feature values
