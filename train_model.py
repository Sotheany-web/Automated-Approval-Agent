import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# THINK LAYER: MODEL TRAINING

print("Training Medical Cost Prediction Model...")

# 1. Load final dataset
try:
    df = pd.read_csv("system_ready_data.csv")
except FileNotFoundError:
    print("ERROR: system_ready_data.csv not found.")
    print("Please run these files first:")
    print("python clean_data.py")
    print("python feature_selection.py")
    sys.exit(1)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns in system_ready_data.csv:")
print(df.columns.tolist())

# 2. Check missing values
if df.isnull().values.any():
    print("ERROR: Dataset contains missing values.")
    print(df.isnull().sum())
    sys.exit(1)

# 3. Define input features and target
target = "Medical_Cost"

if target not in df.columns:
    print(f"ERROR: Target column '{target}' not found.")
    sys.exit(1)

X = df.drop(columns=[target])
y = df[target]

# 4. Define numeric and categorical features
numeric_features = [
    "Age",
    "BMI",
    "Dependents",
    "Smoker",
    "Diabetes"
]

categorical_features = [
    "Sex",
    "Hereditary_Diseases"
]

selected_features = numeric_features + categorical_features

# 5. Check if all selected features exist
missing_features = [col for col in selected_features if col not in X.columns]

if missing_features:
    print("\nERROR: These features are missing from system_ready_data.csv:")
    print(missing_features)
    print("\nPlease check feature_selection.py and train_model.py use the same features.")
    sys.exit(1)

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# 6. Preprocessing
# StandardScaler scales numeric features.
# OneHotEncoder converts categorical features into numeric form.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features],
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 8. Choose regression algorithms
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_model_name = ""
best_test_r2 = -999

# 9. Train and evaluate models
for model_name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("\n==============================")
    print("Model:", model_name)
    print("==============================")

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    mae = mean_absolute_error(y_test, test_pred)
    mse = mean_squared_error(y_test, test_pred)
    rmse = np.sqrt(mse)

    print("Train R2:", train_r2)
    print("Test R2:", test_r2)
    print("Accuracy-like score:", round(test_r2 * 100, 2), "%")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)

    results.append({
        "Model": model_name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Accuracy_Like_Score": round(test_r2 * 100, 2),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_model = pipeline
        best_model_name = model_name

# 10. Save model metrics report
results_df = pd.DataFrame(results)
results_df.to_csv("model_metrics_report.csv", index=False)

# 11. Save best model using joblib
joblib.dump(best_model, "medical_cost_model.pkl")

print("\nSUCCESS: Model training completed!")
print("Best Model:", best_model_name)
print("Best Test R2:", best_test_r2)
print("Accuracy-like score:", round(best_test_r2 * 100, 2), "%")
print("Generated: medical_cost_model.pkl")
print("Generated: model_metrics_report.csv")

print("You may now run: streamlit run app.py")