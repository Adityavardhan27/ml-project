import pandas as pd

df = pd.read_csv("Data/final_output.csv")

print(df["FraudFlag"].value_counts())
