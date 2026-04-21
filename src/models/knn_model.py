def run_knn():

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    print("\n===== KNN =====")

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

    X = df[features].fillna(0)
    y = df["label"]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Handle Imbalance (SMOTE FIRST)
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # Scaling (IMPORTANT FOR KNN)
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Model
    # -----------------------------
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    preds = model.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_knn()