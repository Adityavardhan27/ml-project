import pandas as pd
import numpy as np

df = pd.read_csv("Data/new dataset/if_scored.csv")

# -----------------------------
# StatScore
# -----------------------------
df["StatScore"] = (
    abs(df["Value_z"])
    + abs(df["GasEfficiency_z"])
    + abs(df["TimeGap_z"])
    + abs(df["GasCost_z"])
)

# Normalize StatScore
den1 = df["StatScore"].max() - df["StatScore"].min() + 1e-9
df["StatScore"] = (df["StatScore"] - df["StatScore"].min()) / den1

# -----------------------------
# TempScore
# -----------------------------
rolling_mean = df["Value_z"].rolling(20).mean().fillna(0)
epsilon = 1e-9

df["TempScore"] = abs(df["Value_z"] - rolling_mean) / (rolling_mean + epsilon)

# Normalize TempScore
den2 = df["TempScore"].max() - df["TempScore"].min() + 1e-9
df["TempScore"] = (df["TempScore"] - df["TempScore"].min()) / den2

# -----------------------------
# Final Score
# -----------------------------
df["FinalScore"] = (
    0.3 * df["IF_Score"]
    + 0.4 * df["StatScore"]
    + 0.3 * df["TempScore"]
)

# Threshold (top 15% anomalies)
threshold = df["FinalScore"].quantile(0.85)
df["FraudFlag"] = (df["FinalScore"] > threshold).astype(int)

# Save
df.to_csv("Data/new dataset/final_output.csv", index=False)

print(" MF-UFS completed")