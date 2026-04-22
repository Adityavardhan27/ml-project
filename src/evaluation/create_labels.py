import pandas as pd

df = pd.read_csv("Data/new dataset/final_output.csv", low_memory=False)

# Create label
df["label"] = df["FraudFlag"]

# Keep only useful columns (NO leakage)
keep_cols = [
    "Value_z",
    "GasCost_z",
    "GasEfficiency_z",
    "TimeGap_z",
    "BlockGap_z",
    "IF_Score",
    "StatScore",
    "TempScore",
    "FinalScore",
    "label",
]

df = df[keep_cols]

df.to_csv("Data/new dataset/labeled_data.csv", index=False)

print(" Labels created from MF-UFS")
print(f" Total rows    : {len(df)}")
print(f" Normal  (0)   : {(df['label'] == 0).sum()}")
print(f" Fraud   (1)   : {(df['label'] == 1).sum()}")
print(f" Fraud rate    : {df['label'].mean()*100:.2f}%")