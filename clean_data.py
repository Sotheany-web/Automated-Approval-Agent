import pandas as pd
import numpy as np

# SENSE LAYER PART 1: DATA CLEANING
print("Cleaning Health Insurance Claim Dataset...")

# 1. Load dataset
DATA_FILE = "1651277648862_healthinsurance.csv"
df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# 2. Rename columns from claim to cost
df = df.rename(columns={
    "age": "Age",
    "sex": "Sex",
    "weight": "Weight",
    "bmi": "BMI",
    "hereditary_diseases": "Hereditary_Diseases",
    "no_of_dependents": "Dependents",
    "smoker": "Smoker",
    "city": "City",
    "bloodpressure": "BloodPressure",
    "diabetes": "Diabetes",
    "regular_ex": "Regular_Exercise",
    "job_title": "Job_Title",
    "claim": "Medical_Cost"
})

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Replace infinite values with missing values
df = df.replace([np.inf, -np.inf], np.nan)

# 5. Remove missing values
df = df.dropna()

# 6. Remove invalid records
numeric_columns = [
    "Age", "Weight", "BMI", "Dependents", "Smoker",
    "BloodPressure", "Diabetes", "Regular_Exercise", "Medical_Cost"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df = df[df["Age"] > 0]
df = df[df["Weight"] > 0]
df = df[df["BMI"] > 0]
df = df[df["Dependents"] >= 0]
df = df[df["BloodPressure"] > 0]
df = df[df["Medical_Cost"] >= 0]

# Binary columns should only contain 0 or 1
df = df[df["Smoker"].isin([0, 1])]
df = df[df["Diabetes"].isin([0, 1])]
df = df[df["Regular_Exercise"].isin([0, 1])]

# 7. Ensure correct data types
df["Age"] = df["Age"].astype(int)
df["Weight"] = df["Weight"].astype(float)
df["BMI"] = df["BMI"].astype(float)
df["Dependents"] = df["Dependents"].astype(int)
df["Smoker"] = df["Smoker"].astype(int)
df["BloodPressure"] = df["BloodPressure"].astype(float)
df["Diabetes"] = df["Diabetes"].astype(int)
df["Regular_Exercise"] = df["Regular_Exercise"].astype(int)
df["Medical_Cost"] = df["Medical_Cost"].astype(float)

# 8. Save cleaned dataset before feature selection
df.to_csv("cleaned_before_feature_selection.csv", index=False)

print("\nSUCCESS: cleaned_before_feature_selection.csv created!")
print("Cleaned dataset shape:", df.shape)

print("\nCleaned columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
