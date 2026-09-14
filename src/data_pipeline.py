"""
data_pipeline.py — Sprint 1: Data Preprocessing & Labeling
===========================================================
Module xử lý dữ liệu điểm sinh viên thực tế (Tiếng Việt)
và chuyển đổi sang định dạng chuẩn FIT-HAU phục vụ huấn luyện Decision Tree.
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
RAW_DATA_PATH: str = os.path.join("data", "raw", "student-mat.csv")
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0

# Mở rộng ánh xạ cho tất cả các cột có thể xuất hiện trong file mới của bạn
COLUMN_MAPPING: dict[str, str] = {
    "điểm Toán rời rạc": "Toan_Roi_Rac",
    "điểm kĩ thuật lập trình": "Lap_Trinh_C",
    "điểm kỹ thuật lập trình": "Lap_Trinh_C",
    "điểm CSDL": "Co_So_Du_Lieu",
    "điểm Cơ sở dữ liệu": "Co_So_Du_Lieu",
    "Chuyên Ngành": "Chuyen_Nganh"
}

# Danh sách feature cơ bản (có thể tự động mở rộng theo file thực tế)
DEFAULT_FEATURE_COLUMNS: list[str] = ["Toan_Roi_Rac", "Lap_Trinh_C", "Co_So_Du_Lieu"]
TARGET_COLUMN: str = "Chuyen_Nganh"


# ============================================================
# FUNCTIONS
# ============================================================
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset from possible paths."""
    possible_paths = [
        filepath,
        os.path.join("data", "raw", "student-mat.csv"),
        os.path.join("data", "raw", "DSS_Chuyen_Nganh_Chinh_Xac.csv"),
        "student-mat.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"      --> Tải dữ liệu thành công từ: {path}")
            try:
                df = pd.read_csv(path, encoding='utf-8-sig')
                if len(df.columns) == 1:
                    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
            except Exception:
                df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
            return df

    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu đầu vào. Vui lòng kiểm tra lại đường dẫn file CSV trong thư mục data."
    )


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling missing scores with 0.0 instead of dropping rows."""
    df_cleaned = df.copy()
    score_cols = [col for col in df_cleaned.columns if 'điểm' in col.lower() or 'diem' in col.lower()]

    # Điền giá trị 0.0 cho các môn bị thiếu điểm thay vì xóa dòng
    if score_cols:
        df_cleaned[score_cols] = df_cleaned[score_cols].fillna(0.0)

    # Loại bỏ các dòng hoàn toàn không có tên sinh viên
    for name_col in ['Họ và tên', 'Họ y tên', 'Ho va ten', 'Ten']:
        if name_col in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=[name_col])
            break

    return df_cleaned


def rename_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns and rename to FIT-HAU standard names."""
    df_renamed = df.rename(columns=COLUMN_MAPPING)

    # Lấy tự động tất cả các cột điểm sau khi map hoặc giữ lại các cột số (trừ target)
    # Giúp linh hoạt khi file mới có nhiều cột hơn
    potential_features = [col for col in df_renamed.columns if col != TARGET_COLUMN and col not in ['Họ và tên', 'MSSV', 'STT']]

    # Đảm bảo các cột feature quan trọng luôn tồn tại nếu có
    cols_to_keep = potential_features if len(potential_features) > 0 else DEFAULT_FEATURE_COLUMNS

    if TARGET_COLUMN in df_renamed.columns:
        if TARGET_COLUMN not in cols_to_keep:
            cols_to_keep.append(TARGET_COLUMN)

    # Chỉ lấy những cột thực sự có trong dataframe để tránh lỗi KeyError
    valid_cols = [col for col in cols_to_keep if col in df_renamed.columns]
    return df_renamed[valid_cols].copy()


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all score features are numeric and within 0-10 scale."""
    df_norm = df.copy()
    # Lấy tất cả các cột trừ cột target ra để normalize dạng số điểm
    feature_cols = [col for col in df_norm.columns if col != TARGET_COLUMN]

    for col in feature_cols:
        df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce').fillna(0.0).round(1)
    return df_norm


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign major labels based on normalized scores using vectorized conditions."""
    df_labeled = df.copy()

    if TARGET_COLUMN in df_labeled.columns and not df_labeled[TARGET_COLUMN].isnull().all():
        return df_labeled

    # Xác định các cột để xét điều kiện gán nhãn linh hoạt theo các cột đang có
    col_toan = "Toan_Roi_Rac" if "Toan_Roi_Rac" in df_labeled.columns else df_labeled.columns[0]
    col_laptrinh = "Lap_Trinh_C" if "Lap_Trinh_C" in df_labeled.columns else (df_labeled.columns[1] if len(df_labeled.columns) > 1 else df_labeled.columns[0])

    conditions = [
        (df_labeled[col_toan] < 5.0) | (df_labeled[col_laptrinh] < 5.0),
        (df_labeled[col_toan] >= 6.5) & (df_labeled[col_toan] >= df_labeled[col_laptrinh]),
        (df_labeled[col_laptrinh] >= 6.5)
    ]
    choices = ["Cảnh báo", "AI", "SE"]

    df_labeled[TARGET_COLUMN] = np.select(conditions, choices, default="SE")
    return df_labeled


def run_pipeline() -> pd.DataFrame:
    """Execute the full data pipeline end-to-end."""
    print("=" * 60)
    print("🚀 DSS FIT-HAU — Data Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/5] Loading raw data...")
    df = load_raw_data()
    print(f"      ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Missing values
    print("\n[2/5] Handling missing values...")
    df = handle_missing_values(df)
    print(f"      ✅ Rows retained: {len(df)} (filled missing scores with 0.0)")

    # Step 3: Feature mapping
    print("\n[3/5] Renaming & selecting features...")
    df = rename_and_select_features(df)
    print(f"      ✅ Columns mapped: {list(df.columns)}")

    # Step 4: Normalize
    print("\n[4/5] Normalizing scores...")
    df = normalize_scores(df)
    feature_cols_current = [col for col in df.columns if col != TARGET_COLUMN]
    min_val = df[feature_cols_current].min().min() if feature_cols_current else 0.0
    max_val = df[feature_cols_current].max().max() if feature_cols_current else 0.0
    print(f"      ✅ Score range: {min_val:.1f} — {max_val:.1f}")

    # Step 5: Label
    print("\n[5/5] Assigning major labels...")
    df = assign_labels(df)
    print(f"      ✅ Label distribution:\n{df[TARGET_COLUMN].value_counts().to_string()}")

    # Export
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved → {PROCESSED_DATA_PATH}")
    print("=" * 60)

    return df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_pipeline()