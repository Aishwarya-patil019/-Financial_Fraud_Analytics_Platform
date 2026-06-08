import pandas as pd
import os

# Dataset path
DATA_PATH = "/workspaces/-Financial_Fraud_Analytics_Platform/data/Fraud Detection Dataset.csv"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Display basic information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Records: {duplicates}")

df = df.drop_duplicates()

print(f"Dataset Shape After Removing Duplicates: {df.shape}")

# Create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
output_path = "data/processed/cleaned_creditcard.csv"
df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
print("Preprocessing completed successfully.")