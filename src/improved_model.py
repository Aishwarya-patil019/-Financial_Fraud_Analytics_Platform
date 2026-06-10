import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("data/processed/cleaned_creditcard.csv")

# Handle missing values
df = df.ffill()

# Encode categorical columns
le = LabelEncoder()

categorical_columns = [
    "Transaction_ID",
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Fraudulent", axis=1)
y = df["Fraudulent"]

# Balance dataset
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("Balanced Dataset Shape:")
print(y_resampled.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Train model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
predictions = rf.predict(X_test)

# Metrics
print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(rf, "models/fraud_model.pkl")

print("\nModel saved successfully.")