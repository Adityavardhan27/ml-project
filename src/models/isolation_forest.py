import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("Data/new dataset/processed_data.csv")

features = ["Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z", "BlockGap_z"]
X = df[features]

model = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
model.fit(X)

scores = model.decision_function(X)

# Normalize safely
den = scores.max() - scores.min() + 1e-9
df["IF_Score"] = (scores.max() - scores) / den

# Binary anomaly label (for reference only)
df["IF_Label"] = model.predict(X)
df["IF_Label"] = df["IF_Label"].map({1: 0, -1: 1})

# Save
df_final = df[features + ["IF_Score", "IF_Label"]]
df_final.to_csv("Data/new dataset/if_scored.csv", index=False)

print(" Isolation Forest done")