import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Function to preprocess data (handle missing values, encode categorical features)
def preprocess_data(data):
  # Handle missing values (optional, you can explore more sophisticated methods)
  data.dropna(axis=1, inplace=True)  # Drop columns with missing values

  # Encode categorical features (stress level)
  encoder = LabelEncoder()
  data["stress_level"] = encoder.fit_transform(data["stress_level"])
  return data, encoder

# Load the stress level dataset
data, encoder = preprocess_data(pd.read_csv("StressLevelDataset.csv"))  # Replace with your CSV path

# Separate features (X) and target variable (y)
X = data.drop("stress_level", axis=1)
y = data["stress_level"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier model
tree_clf = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_clf.fit(X_train, y_train)

# Save the trained model and encoder using pickle
with open('stress_model.pkl', 'wb') as f:
    pickle.dump(tree_clf, f)
    pickle.dump(encoder, f)

print("Model training complete and saved as model.pkl")
