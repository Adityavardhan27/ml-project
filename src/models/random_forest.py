def run_random_forest():
    import pandas as pd
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    from imblearn.over_sampling import SMOTE

    print("\n===== RANDOM FOREST (Final Clean Version) =====")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv("Data/new dataset/labeled_data.csv")

    # -----------------------------
    # FEATURES (NO LEAKAGE)
    # -----------------------------
    features = [
        "Value_z",
        "GasCost_z",
        "GasEfficiency_z",
        "TimeGap_z",
        "BlockGap_z"
    ]

    X = df[features].fillna(0)
    y = df["FraudFlag"]

    # -----------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # SMOTE (NO SCALING NEEDED)
    # -----------------------------
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # MODEL (STRONG CONFIG)
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -----------------------------
    # REPORT
    # -----------------------------
    print("\n Accuracy:", accuracy_score(y_test, preds))
    print("\n Precision:", precision_score(y_test, preds))
    print(" Recall   :", recall_score(y_test, preds))
    print(" F1 Score :", f1_score(y_test, preds))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n Classification Report:")
    print(classification_report(y_test, preds))

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    os.makedirs("Models", exist_ok=True)
    joblib.dump(model, "Models/random_forest.pkl")

    print("\nModel saved")

    return model


if __name__ == "__main__":
    run_random_forest()