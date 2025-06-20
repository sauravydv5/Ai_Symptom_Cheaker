



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("train_disease.csv")

# Separate features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and feature names as a tuple
joblib.dump((model, X.columns.tolist()), "disease_model.pkl")

print("✅ Model trained and saved as 'disease_model.pkl'")
# cols = pd.read_csv("train_disease.csv").drop("prognosis", axis=1).columns.tolist()
# print(repr(cols))
