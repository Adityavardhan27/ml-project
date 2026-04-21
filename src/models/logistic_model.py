def run_logistic():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv("Data/labeled_data.csv")

    # -----------------------------
    # Feature Engineering (NEW DATA)
    # -----------------------------
    df["value_ratio"] = df["Value"] / (df["GasCost"] + 1)
    df["gas_efficiency"] = df["GasEfficiency"]
    df["is_high_value"] = (df["Value"] > df["Value"].mean()).astype(int)

    # -----------------------------
    # Features
    # -----------------------------
    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z",
        "value_ratio",
        "gas_efficiency",
        "is_high_value"
    ]

    X = df[features]
    y = df["label"]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Handle Imbalance (SMOTE)
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # Model
    # -----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_logistic()