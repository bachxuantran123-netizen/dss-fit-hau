"""
train_core.py — Sprint 1: AI Core Training Pipeline
===================================================
Module huấn luyện mô hình Decision Tree tự động cho hệ thống DSS FIT-HAU,
hỗ trợ linh hoạt cấu trúc dữ liệu mở rộng với 23 features.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONSTANTS
# ============================================================
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")
MODEL_SAVE_PATH: str = os.path.join("models", "decision_tree_model.pkl")

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
    """Khởi tạo và huấn luyện mô hình Decision Tree Classifier."""
    clf = DecisionTreeClassifier(random_state=42, max_depth=6)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Đánh giá hiệu suất mô hình trên tập test."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"      ✅ Test Accuracy: {acc * 100:.2f}%")
    print("\n[4/5] Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-" * 50)
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