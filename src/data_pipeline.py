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

# Ánh xạ tên cột từ Tiếng Việt sang Tiếng Anh chuẩn cho mô hình train
COLUMN_MAPPING: dict[str, str] = {
    "điểm Toán rời rạc": "Toan_Roi_Rac",
    "điểm kĩ thuật lập trình": "Lap_Trinh_C",
    "điểm CSDL": "Co_So_Du_Lieu",
    "Chuyên Ngành": "Chuyen_Nganh"
}

FEATURE_COLUMNS: list[str] = ["Toan_Roi_Rac", "Lap_Trinh_C", "Co_So_Du_Lieu"]
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
    score_cols = [col for col in df_cleaned.columns if 'điểm' in col]

    # Điền giá trị 0.0 cho các môn bị thiếu điểm thay vì xóa dòng
    df_cleaned[score_cols] = df_cleaned[score_cols].fillna(0.0)

    # Loại bỏ các dòng hoàn toàn không có tên sinh viên
    if 'Họ và tên' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['Họ y tên'] if 'Họ y tên' in df_cleaned.columns else ['Họ và tên'])

    return df_cleaned


def rename_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns and rename to FIT-HAU standard names."""
    df_renamed = df.rename(columns=COLUMN_MAPPING)

    cols_to_keep = [col for col in FEATURE_COLUMNS if col in df_renamed.columns]
    if TARGET_COLUMN in df_renamed.columns:
        cols_to_keep.append(TARGET_COLUMN)

    return df_renamed[cols_to_keep].copy()


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all score features are numeric and within 0-10 scale."""
    df_norm = df.copy()
    for col in FEATURE_COLUMNS:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce').fillna(0.0).round(1)
    return df_norm


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign major labels based on normalized scores using vectorized conditions."""
    df_labeled = df.copy()

    if TARGET_COLUMN in df_labeled.columns and not df_labeled[TARGET_COLUMN].isnull().all():
        return df_labeled

    conditions = [
        (df_labeled["Toan_Roi_Rac"] < 5.0) | (df_labeled["Lap_Trinh_C"] < 5.0),
        (df_labeled["Toan_Roi_Rac"] >= 6.5) & (df_labeled["Toan_Roi_Rac"] >= df_labeled["Lap_Trinh_C"]),
        (df_labeled["Lap_Trinh_C"] >= 6.5)
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
    rows_before = len(df)
    df = handle_missing_values(df)
    print(f"      ✅ Rows retained: {len(df)} (filled missing scores with 0.0)")

    # Step 3: Feature mapping
    print("\n[3/5] Renaming & selecting features...")
    df = rename_and_select_features(df)
    print(f"      ✅ Columns mapped: {list(df.columns)}")

    # Step 4: Normalize
    print("\n[4/5] Normalizing scores...")
    df = normalize_scores(df)
    print(f"      ✅ Score range: {df[FEATURE_COLUMNS].min().min():.1f} — {df[FEATURE_COLUMNS].max().max():.1f}")

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