"""
data_pipeline.py — Data Preprocessing & Labeling (v2)
======================================================
Module xử lý dữ liệu điểm sinh viên FIT-HAU từ file CSV đã trích xuất từ PDF.
Gán nhãn 5 hướng nghề nghiệp dựa trên Ma trận Kỹ năng (Skill Matrix).

Pipeline: FIT_HAU_Raw_Scores.csv → Clean → Label → FIT_HAU_Cleaned.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
RAW_DATA_PATH: str = os.path.join("data", "raw", "FIT_HAU_Raw_Scores.csv")
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")

TARGET_COLUMN: str = "Chuyen_Nganh"

# Metadata columns — sẽ bị loại bỏ khỏi features trước khi train
NON_FEATURE_COLUMNS: list[str] = ["Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"]

# ============================================================
# SKILL MATRIX — Ánh xạ nhóm môn học → Hướng nghề nghiệp
# ============================================================
# Đây là phần có thể thay thế bằng K-Means clustering sau này.
# Hiện tại dùng Heuristic Rules (Expert-defined Skill Matrix).
SKILL_MATRIX: dict[str, list[str]] = {
    "Software Engineer": [
        "CÔNG NGHỆ PHẦN MỀM",
        "KỸ THUẬT LẬP TRÌNH",
        "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
        "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
        "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    ],
    "Data Engineer": [
        "CƠ SỞ DỮ LIỆU",
        "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    ],
    "AI Engineer": [
        "TOÁN RỜI RẠC",
        "XỬ LÝ TÍN HIỆU SỐ",
        "XỬ LÝ ẢNH",
        "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    ],
    "Security Engineer": [
        "AN NINH MẠNG",
        "AN TOÀN VÀ BẢO MẬT HTTT",
        "MẠNG MÁY TÍNH",
        "QUẢN TRỊ MẠNG MÁY TÍNH",
    ],
    "System/DevOps": [
        "HỆ ĐIỀU HÀNH",
        "HỆ ĐIỀU HÀNH LINUX",
        "KIẾN TRÚC MÁY TÍNH",
    ],
}

DEFAULT_LABEL: str = "Software Engineer"


# ============================================================
# FUNCTIONS
# ============================================================
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset extracted from PDFs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu tại: {filepath}. "
            f"Hãy chạy pdf_extractor.py trước."
        )
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: ensure numeric scores, fill NaN with 0.0."""
    df_clean = df.copy()

    # Identify score columns (all except metadata)
    score_cols = [c for c in df_clean.columns if c not in NON_FEATURE_COLUMNS]

    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0).round(1)

    return df_clean


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign career path labels based on Skill Matrix.

    Logic:
        1. For each student, compute the mean score for each career group.
        2. The group with the highest mean score (and > 0) gets assigned.
        3. If all groups have mean 0 → assign DEFAULT_LABEL.

    NOTE: This function is MODULAR — you can replace the logic below
    with K-Means clustering in the future without changing any other file.
    """
    df_labeled = df.copy()
    labels = []

    # Pre-filter: only use columns that actually exist in the DataFrame
    valid_matrix: dict[str, list[str]] = {}
    for career, subjects in SKILL_MATRIX.items():
        valid_subjects = [s for s in subjects if s in df_labeled.columns]
        if valid_subjects:
            valid_matrix[career] = valid_subjects

    for _, row in df_labeled.iterrows():
        best_career = DEFAULT_LABEL
        best_avg = 0.0

        for career, subjects in valid_matrix.items():
            scores = [row[s] for s in subjects if row[s] > 0]
            if scores:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_career = career

        labels.append(best_career)

    df_labeled[TARGET_COLUMN] = labels
    return df_labeled


def prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Remove metadata columns, keep only score features + target label."""
    cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df_final = df.drop(columns=cols_to_drop)
    return df_final


def run_pipeline() -> pd.DataFrame:
    """Execute the full data pipeline end-to-end."""
    print("=" * 60)
    print("🚀 DSS FIT-HAU — Data Pipeline v2")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/4] Loading raw data...")
    df = load_raw_data()
    print(f"      ✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Clean
    print("\n[2/4] Cleaning data...")
    df = clean_data(df)
    score_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    non_zero_count = (df[score_cols] > 0).any(axis=1).sum()
    print(f"      ✅ Students with at least 1 score > 0: {non_zero_count}/{len(df)}")

    # Step 3: Label
    print("\n[3/4] Assigning career labels (Skill Matrix)...")
    df = assign_labels(df)
    print(f"      ✅ Label distribution:")
    print(df[TARGET_COLUMN].value_counts().to_string(header=False))

    # Step 4: Prepare for training (remove metadata)
    print("\n[4/4] Preparing for training...")
    df_final = prepare_for_training(df)
    feature_cols = [c for c in df_final.columns if c != TARGET_COLUMN]
    print(f"      ✅ Features: {len(feature_cols)} subjects")
    print(f"      ✅ Target: {TARGET_COLUMN}")

    # Export
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Saved → {PROCESSED_DATA_PATH}")
    print(f"  📐 Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print("=" * 60)

    return df_final


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_pipeline()