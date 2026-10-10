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

# Phiên bản mô hình hiện tại
MODEL_VERSION: str = "1.0.0"

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")
MODEL_SAVE_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")
FEATURE_IMPORTANCE_PATH: str = os.path.join("reports", "dss_feature_importance.png")

TARGET_COLUMN: str = "Chuyen_Nganh"


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


def split_data(df: pd.DataFrame) -> tuple:
    """Chia tập dữ liệu thành train/test tự động dựa trên các cột thực tế có trong df.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_columns)
    """
    # Xác định tất cả các cột làm feature (loại bỏ cột target)
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    if not feature_columns:
        raise ValueError("Không tìm thấy cột feature nào trong DataFrame sau khi loại bỏ cột target.")

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # Chia train/test (80% / 20%), giữ nguyên tỷ lệ nhãn bằng stratify nếu có thể
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        # Trường hợp một số lớp quá ít mẫu không thể stratify
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Khởi tạo và huấn luyện mô hình Decision Tree Classifier với GridSearchCV."""
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, None],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    base_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"      ✅ Best parameters: {grid_search.best_params_}")
    print(f"      ✅ Best CV Score: {grid_search.best_score_:.4f}")
    print(f"      ✅ CV Score Std: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")
    return grid_search.best_estimator_


def evaluate_model(clf: DecisionTreeClassifier, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, feature_columns: list[str]):
    """Đánh giá hiệu suất mô hình trên tập test và lưu các biểu đồ."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # So sánh Train Accuracy vs Test Accuracy để phát hiện overfitting
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"      ✅ Train Accuracy: {train_acc * 100:.2f}%")
    print(f"      ✅ Test Accuracy: {acc * 100:.2f}%")
    print(f"      ✅ Gap (Train - Test): {(train_acc - acc) * 100:.2f}%")
    if (train_acc - acc) < 0.05:
        print("      ✅ Mô hình KHÔNG bị overfitting (gap < 5%)")
    else:
        print("      ⚠️  Mô hình CÓ THỂ bị overfitting (gap >= 5%)")

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
    plot_tree(clf, feature_names=feature_columns, class_names=clf.classes_, filled=True, rounded=True, ax=ax, fontsize=10, max_depth=3)
    plt.title("Cấu trúc Cây Quyết Định (Top 3 Levels)")
    plt.tight_layout()
    plt.savefig(TREE_PLOT_PATH, dpi=300)
    plt.close()
    print(f"      💾 Saved Tree Plot -> {TREE_PLOT_PATH}")

    # 3. Sinh và lưu Feature Importance Bar Chart
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(np.array(feature_columns)[sorted_idx], importances[sorted_idx])
    plt.title("Feature Importance (Gini/Entropy Decrease)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300)
    plt.close()
    print(f"      💾 Saved Feature Importance -> {FEATURE_IMPORTANCE_PATH}")

    return acc


def save_model(clf: DecisionTreeClassifier, filepath: str = MODEL_SAVE_PATH):
    """Lưu mô hình đã huấn luyện ra file pkl dưới dạng dictionary 4 khoá.
    
    Dictionary gồm:
        - model: Đối tượng DecisionTreeClassifier đã huấn luyện
        - features: Danh sách tên feature (môn học)
        - classes: Danh sách tên lớp (chuyên ngành)
        - version: Phiên bản mô hình để kiểm tra tương thích
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_artifact = {
        'model': clf,
        'features': list(clf.feature_names_in_),
        'classes': list(clf.classes_),
        'version': MODEL_VERSION
    }
    joblib.dump(model_artifact, filepath)
    print(f"      💾 Saved model (dict 4 keys) -> {filepath}")
    print(f"         Keys: {list(model_artifact.keys())}")
    print(f"         Version: {MODEL_VERSION}")


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
    X_train, X_test, y_train, y_test, feature_columns = split_data(df)
    print(f"      ✅ Features used: {len(feature_columns)} columns")
    print(f"      ✅ Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 3: Train model
    print("\n[3/5] Training Decision Tree Classifier...")
    clf = train_model(X_train, y_train)
    print("      ✅ Training completed successfully.")

    # Step 4: Evaluate model
    print("\n[4/5] Evaluating model performance...")
    evaluate_model(clf, X_train, y_train, X_test, y_test, feature_columns)

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