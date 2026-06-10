import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Example transaction
sample_transaction = {
    "Transaction_ID": 100,
    "User_ID": 4174,
    "Transaction_Amount": 2500,
    "Transaction_Type": 1,
    "Time_of_Transaction": 14,
    "Device_Used": 1,
    "Location": 5,
    "Previous_Fraudulent_Transactions": 0,
    "Account_Age": 100,
    "Number_of_Transactions_Last_24H": 10,
    "Payment_Method": 2
}

# Convert to DataFrame
input_data = pd.DataFrame([sample_transaction])

# Prediction
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("⚠ Fraudulent Transaction Detected")
else:
    print("✓ Legitimate Transaction")