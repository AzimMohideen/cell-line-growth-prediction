import pandas as pd
import os

# Load dataset
df = pd.read_excel("../data/Cell_Lines_Details.xlsx")

# Clean column names (very important)
df.columns = df.columns.str.strip()

print("Available columns:")
print(df.columns.tolist())

# Try to auto-detect Growth column
possible_targets = [
    "GrowthProperties",
    "Growth Properties",
    "GrowthProperty",
    "Growth"
]

TARGET = None
for col in possible_targets:
    if col in df.columns:
        TARGET = col
        break

if TARGET is None:
    raise ValueError(
        "Growth column not found. Update TARGET manually from printed columns."
    )

print("Using target column:", TARGET)

# Drop empty columns
df = df.dropna(axis=1, how="all")

# Drop rows without target
df = df.dropna(subset=[TARGET])

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Final dataset
final_df = pd.concat([X_encoded, y], axis=1)

# Save processed dataset
os.makedirs("../data", exist_ok=True)
final_df.to_csv("../data/processed.csv", index=False)

print("Preprocessing complete.")
print("Final shape:", final_df.shape)
