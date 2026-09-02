"""
train_core.py — Sprint 2: AI Core Training Module
====================================================
Module huấn luyện Decision Tree Classifier với GridSearchCV
và xuất model artifact (.pkl) + báo cáo đánh giá (.png).

Pipeline:
    1. Load cleaned data (FIT_HAU_Cleaned.csv)
    2. Train/Test split (80-20, random_state=42)
    3. GridSearchCV — tìm max_depth tối ưu
    4. Evaluate: Accuracy + Confusion Matrix heatmap
    5. Visualize: Decision Tree plot
    6. Export: dss_brain.pkl (Joblib)
"""

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ============================================================
# CONSTANTS
# ============================================================
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")
MODEL_OUTPUT_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

FEATURE_COLUMNS: list[str] = ["Toan_Roi_Rac", "Lap_Trinh_C", "Co_So_Du_Lieu"]
TARGET_COLUMN: str = "Chuyen_Nganh"

# GridSearchCV parameter grid
PARAM_GRID: dict = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


# ============================================================
# FUNCTIONS
# ============================================================
def load_processed_data(filepath: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned dataset for training."""
    # TODO: Sprint 2 - Task 1 (part 1)
    raise NotImplementedError("Sprint 2: Implement data loading")


def split_data(df: pd.DataFrame) -> tuple:
    """Split data into train/test sets (80-20).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: Sprint 2 - Task 1
    raise NotImplementedError("Sprint 2 - Task 1: Implement train/test split")


def train_with_gridsearch(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train DecisionTreeClassifier with GridSearchCV hyperparameter tuning.

    Must use cross-validation (cv=CV_FOLDS).
    Must report best parameters found.

    Returns:
        DecisionTreeClassifier: Best estimator from GridSearchCV.
    """
    # TODO: Sprint 2 - Task 2
    raise NotImplementedError("Sprint 2 - Task 2: Implement GridSearchCV training")


def evaluate_model(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model accuracy and save Confusion Matrix heatmap.

    Saves: reports/dss_confusion_matrix.png
    Returns: Accuracy score (float)
    """
    # TODO: Sprint 2 - Task 3
    raise NotImplementedError("Sprint 2 - Task 3: Implement evaluation + confusion matrix")


def plot_decision_tree(model: DecisionTreeClassifier) -> None:
    """Visualize and save the Decision Tree structure.

    Saves: reports/dss_tree.png
    """
    # TODO: Sprint 2 - Task 4
    raise NotImplementedError("Sprint 2 - Task 4: Implement tree visualization")


def export_model(model: DecisionTreeClassifier, filepath: str = MODEL_OUTPUT_PATH) -> None:
    """Serialize trained model to .pkl using Joblib.

    Saves: models/dss_brain.pkl
    """
    # TODO: Sprint 2 - Task 5
    raise NotImplementedError("Sprint 2 - Task 5: Implement model export")


def run_training_pipeline() -> None:
    """Execute the full training pipeline end-to-end."""
    print("=" * 60)
    print("🧠 DSS FIT-HAU — AI Core Training Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading processed data...")
    df = load_processed_data()
    print(f"      ✅ Dataset: {df.shape[0]} samples, {df.shape[1]} features")

    # Step 2: Split
    print("\n[2/5] Splitting train/test ({:.0%}/{:.0%})...".format(1 - TEST_SIZE, TEST_SIZE))
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      ✅ Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 3: Train
    print(f"\n[3/5] Training with GridSearchCV (cv={CV_FOLDS})...")
    best_model = train_with_gridsearch(X_train, y_train)
    print(f"      ✅ Best params: {best_model.get_params()}")

    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    accuracy = evaluate_model(best_model, X_test, y_test)
    print(f"      ✅ Accuracy: {accuracy:.2%}")

    # Step 5: Visualize
    print("\n[5/5] Generating reports...")
    plot_decision_tree(best_model)
    print(f"      ✅ Tree plot → {TREE_PLOT_PATH}")

    # Export
    export_model(best_model)
    print(f"\n💾 Model saved → {MODEL_OUTPUT_PATH}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_training_pipeline()
