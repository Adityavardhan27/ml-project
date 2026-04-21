import pandas as pd

df = pd.read_csv("Data/final_output.csv")

df["label"] = df["FraudFlag"]

df.to_csv("Data/labeled_data.csv", index=False)

print(" Labels created from MF-UFS")