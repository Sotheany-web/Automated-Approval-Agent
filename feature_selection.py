import pandas as pd

# SENSE LAYER PART 2: AUTOMATIC FEATURE SELECTION
print("Running Automatic Feature Selection...")

# 1. Load cleaned dataset
df = pd.read_csv("cleaned_before_feature_selection.csv")

print("Cleaned dataset shape:", df.shape)

target = "Medical_Cost"

if target not in df.columns:
    print(f"ERROR: Target column '{target}' not found.")
    exit()

feature_decisions = []
categorical_impact_rows = []

# Feature selection thresholds
NUMERIC_KEEP_THRESHOLD = 0.10
NUMERIC_REVIEW_THRESHOLD = 0.08

CATEGORICAL_KEEP_THRESHOLD = 1000
CATEGORICAL_REVIEW_THRESHOLD = 500

# Numeric feature selection
numeric_features = [
    "Age",
    "Weight",
    "BMI",
    "Dependents",
    "Smoker",
    "BloodPressure",
    "Diabetes",
    "Regular_Exercise"
]

selected_numeric_features = []
review_features = []
removed_features = []

for feature in numeric_features:

    correlation = df[feature].corr(df[target])
    abs_correlation = abs(correlation)

    if abs_correlation >= NUMERIC_KEEP_THRESHOLD:
        decision = "Keep"
        selected_numeric_features.append(feature)

    elif abs_correlation >= NUMERIC_REVIEW_THRESHOLD:
        decision = "Review"
        review_features.append(feature)

    else:
        decision = "Remove"
        removed_features.append(feature)

    feature_decisions.append({
        "Feature": feature,
        "Feature_Type": "Numeric",
        "Method": "Pearson correlation",
        "Score": round(correlation, 6),
        "Abs_Score": round(abs_correlation, 6),
        "Decision": decision
    })

# catergorical feature selection
categorical_features = [
    "Sex",
    "Hereditary_Diseases",
    "City",
    "Job_Title"
]

selected_categorical_features = []

overall_mean = df[target].mean()

for feature in categorical_features:

    category_stats = df.groupby(feature)[target].agg(
        ["mean", "count"]
    ).reset_index()

    category_stats["Difference_From_Overall_Mean"] = (
        category_stats["mean"] - overall_mean
    )

    category_stats["Absolute_Difference"] = (
        category_stats["Difference_From_Overall_Mean"].abs()
    )

    max_difference = category_stats["Absolute_Difference"].max()

    for _, row in category_stats.iterrows():
        categorical_impact_rows.append({
            "Feature": feature,
            "Category": row[feature],
            "Average_Medical_Cost": round(row["mean"], 2),
            "Count": int(row["count"]),
            "Difference_From_Overall_Mean": round(row["Difference_From_Overall_Mean"], 2),
            "Absolute_Difference": round(row["Absolute_Difference"], 2)
        })

    if max_difference >= CATEGORICAL_KEEP_THRESHOLD:
        decision = "Keep"
        selected_categorical_features.append(feature)

    elif max_difference >= CATEGORICAL_REVIEW_THRESHOLD:
        decision = "Review"
        review_features.append(feature)

    else:
        decision = "Remove"
        removed_features.append(feature)

    feature_decisions.append({
        "Feature": feature,
        "Feature_Type": "Categorical",
        "Method": "Average medical cost by category",
        "Score": round(max_difference, 2),
        "Decision": decision
    })
#save reports to csv
feature_report = pd.DataFrame(feature_decisions)
feature_report.to_csv("feature_selection_report.csv", index=False)

categorical_impact = pd.DataFrame(categorical_impact_rows)
categorical_impact.to_csv("categorical_feature_impact.csv", index=False)

print("\nFeature selection report:")
print(feature_report)

print("\nSelected numeric features:")
print(selected_numeric_features)

print("\nSelected categorical features:")
print(selected_categorical_features)

print("\nReview features:")
print(review_features)

print("\nRemoved features:")
print(removed_features)

print("\nSUCCESS: feature_selection_report.csv created!")
print("SUCCESS: categorical_feature_impact.csv created!")

# Include both Keep and Review features.
final_features = []

for row in feature_decisions:
    if row["Decision"] in ["Keep", "Review"]:
        final_features.append(row["Feature"])

# Add target column
final_features.append(target)

# Remove duplicate feature names if any
final_features = list(dict.fromkeys(final_features))

final_df = df[final_features].copy()

final_df.to_csv("system_ready_data.csv", index=False)

print("\nSUCCESS: system_ready_data.csv created!")
print("Final dataset shape:", final_df.shape)

print("\nFinal selected features:")
print(final_features)

print("\nFirst 5 rows:")
print(final_df.head())