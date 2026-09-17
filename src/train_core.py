"""
train_core.py — AI Core Training Pipeline (Tầng 2: Decision Tree)
==================================================================
Module huấn luyện mô hình Decision Tree tự động cho hệ thống DSS FIT-HAU.

Kiến trúc Hybrid 2 tầng:
    Tầng 1 (data_pipeline.py): Content-Based Filtering (Cosine Similarity)
        → Gán nhãn cho dữ liệu training.
    Tầng 2 (file này): Decision Tree (GridSearchCV)
        → Học từ nhãn đã gán, dự đoán cho user mới, cung cấp XAI.

Hỗ trợ linh hoạt cấu trúc dữ liệu mở rộng với 20 features (môn học).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")
MODEL_SAVE_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")

TARGET_COLUMN: str = "Chuyen_Nganh"
FEATURE_COLUMNS: list[str] = []  # Sẽ được tự động cập nhật dựa trên dữ liệu thực tế


# ============================================================
# FUNCTIONS
# ============================================================
def load_processed_data(filepath: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned dataset from processed folder."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu đã xử lý tại: {filepath}. Vui lòng chạy data_pipeline.py trước."
        )
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def split_data(df: pd.DataFrame):
    """Chia tập dữ liệu thành train/test tự động dựa trên các cột thực tế có trong df."""
    global FEATURE_COLUMNS
    # Xác định tất cả các cột làm feature (loại bỏ cột target)
    FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]

    if not FEATURE_COLUMNS:
        raise ValueError("Không tìm thấy cột feature nào trong DataFrame sau khi loại bỏ cột target.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Chia train/test (80% / 20%), giữ nguyên tỷ lệ nhãn bằng stratify nếu có thể
    try:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        # Trường hợp một số lớp quá ít mẫu không thể stratify
        return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Khởi tạo và huấn luyện mô hình Decision Tree Classifier với GridSearchCV."""
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, None],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10]
    }
    
    base_clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"      ✅ Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model(clf: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Đánh giá hiệu suất mô hình trên tập test và lưu các biểu đồ."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"      ✅ Test Accuracy: {acc * 100:.2f}%")
    print("\n[4/5] Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-" * 50)
    
    # Tạo thư mục reports nếu chưa tồn tại
    os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
    
    # 1. Sinh và lưu Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, cmap='Blues')
    plt.xticks(rotation=45, ha="right")  # Xoay nhãn trục x để không bị đè lên nhau
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"      💾 Saved Confusion Matrix -> {CONFUSION_MATRIX_PATH}")

    # 2. Sinh và lưu Sơ đồ cây (Tree Plot)
    fig, ax = plt.subplots(figsize=(20, 10))
    # Giới hạn max_depth=3 để cây không bị rối rắm trên hình vẽ
    plot_tree(clf, feature_names=FEATURE_COLUMNS, class_names=clf.classes_, filled=True, rounded=True, ax=ax, fontsize=10, max_depth=3)
    plt.title("Cấu trúc Cây Quyết Định (Top 3 Levels)")
    plt.tight_layout()
    plt.savefig(TREE_PLOT_PATH, dpi=300)
    plt.close()
    print(f"      💾 Saved Tree Plot -> {TREE_PLOT_PATH}")

    return acc


def save_model(clf: DecisionTreeClassifier, filepath: str = MODEL_SAVE_PATH):
    """Lưu mô hình đã huấn luyện ra file pkl."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(clf, filepath)
    print(f"      💾 Saved model -> {filepath}")


def run_training_pipeline() -> DecisionTreeClassifier:
    """Thực thi toàn bộ pipeline huấn luyện mô hình end-to-end."""
    print("=" * 60)
    print("🧠 DSS FIT-HAU — AI Core Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading processed data...")
    df = load_processed_data()
    print(f"      ✅ Dataset: {len(df)} samples, {len(df.columns)} columns")

    # Step 2: Split data
    print("\n[2/5] Splitting train/test (80%/20%)...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      ✅ Features used: {len(FEATURE_COLUMNS)} columns")
    print(f"      ✅ Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 3: Train model
    print("\n[3/5] Training Decision Tree Classifier...")
    clf = train_model(X_train, y_train)
    print("      ✅ Training completed successfully.")

    # Step 4: Evaluate model
    print("\n[4/5] Evaluating model performance...")
    evaluate_model(clf, X_test, y_test)

    # Step 5: Save model
    print("\n[5/5] Saving trained model artifact...")
    save_model(clf)

    print("=" * 60)
    print("🎉 Pipeline training completed successfully!")
    print("=" * 60)
    return clf


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_training_pipeline()