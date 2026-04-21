print("\n===== STARTING FULL PIPELINE =====\n")

print("1. Running Feature Engineering...")
import src.preprocessing.feature_engineering

print("2. Running Isolation Forest...")
import src.models.isolation_forest

print("3. Running MF-UFS Algorithm...")
import src.models.mfufs

print("4. Creating Labels...")
import src.evaluation.create_labels

from src.models.logistic_model import run_logistic
from src.models.svm_model import run_svm
from src.models.knn_model import run_knn
from src.models.decision_tree_model import run_decision_tree


print("5. Running Logistic Regression...")
run_logistic()

print("6. Running SVM...")
run_svm()

print("7. Running KNN...")
run_knn()

print("8. Running Decision Tree...")
run_decision_tree()

print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")