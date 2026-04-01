def run_knn():

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler

    print("\n===== KNN =====")

    # 1️⃣ Load data
    df = pd.read_csv("Data/labeled_data.csv")

    features = [
        "Total Value_z",
        "Net Value_z",
        "Fee Ratio_z",
        "Time Gap_z",
        "Block Gap_z"
    ]

    X = df[features]
    y = df["label"]

    # 2️⃣ Train-test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3️⃣ Scaling (MANDATORY for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4️⃣ Model (slightly tuned)
    model = KNeighborsClassifier(n_neighbors=7)  # 🔥 try 7 instead of 5
    model.fit(X_train, y_train)

    # 5️⃣ Predictions
    preds = model.predict(X_test)

    # 6️⃣ Evaluation
    print(classification_report(y_test, preds))
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))


if __name__ == "__main__":
    run_knn()