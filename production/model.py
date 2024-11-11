# %% [markdown]
# # Human Activity Recognition model

# %%

import mlflow
mlflow.autolog()

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()


# %% [markdown]
# ## load and read data 

try:
    data = pd.read_csv(args.trainingdata)
except pd.errors.ParserError:
    data = pd.read_csv(args.trainingdata, error_bad_lines=False)

# %%
# Load the training and test data
# data = pd.read_csv('../activity_recognition.csv')  # Path to the training data
# test_data = pd.read_csv('../test.csv')    # Path to the test data

# %% [markdown]
# ## Separate features and labels

# %%
# Separate features and target label
X = data.drop('Activity', axis=1)
y = data['Activity']

# %% [markdown]
# ## Split the data into training and testing sets

# %%
# Split the merged dataset into new training and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [markdown]
# ## Standization

# %%
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Train a Random Forest classifier

# %%
# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate model

# %%
# Make predictions
y_pred = clf.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


