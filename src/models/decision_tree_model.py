import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def run_decision_tree():
    print("\n===== DECISION TREE MODEL (on final_output.csv) =====")
    
    df = pd.read_csv("Data/final_output.csv")
    
    if all(col in df.columns for col in ["Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z"]):
        features = ["Value_z", "GasCost_z", "GasEfficiency_z", "TimeGap_z"]
        print("Using z-scored features.")
    else:
        features = ["Value", "GasCost", "GasEfficiency", "TimeGap"]
        print("Using raw features.")
    
    X = df[features]
    y = df["FraudFlag"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Model instantiation
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    # Train
    dt.fit(X_train, y_train)
    
    # Evaluate
    y_pred = dt.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # Save model
    os.makedirs("Models", exist_ok=True)
    joblib.dump(dt, "Models/decision_tree.pkl")
    print("Decision Tree model saved to Models/decision_tree.pkl")
    
    return dt
if _name_ == "__main__":
    run_decision_tree()