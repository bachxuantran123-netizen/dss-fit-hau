"""
merge_new_data.py — Merge file điểm mới vào FIT_HAU_Raw_Scores.csv
====================================================================
Nạp file Danh_Sach_70_Sinh_Vien_Diem.csv (4 môn mới: Java, C#, Web, Trí tuệ nhân tạo),
ánh xạ cột, merge vào dữ liệu gốc 1256 SV.

Output cập nhật: data/raw/FIT_HAU_Raw_Scores.csv
"""

import os
import sys
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
NEW_DATA_PATH: str = os.path.join("data", "raw", "Danh_Sach_70_Sinh_Vien_Diem.csv")
RAW_CSV_PATH: str = os.path.join("data", "raw", "FIT_HAU_Raw_Scores.csv")

# Mapping: column name in new CSV → standardized column name
NEW_SUBJECT_MAPPING: dict[str, str] = {
    "Java": "JAVA",
    "C#": "C#",
    "Web": "LẬP TRÌNH WEB",
    "Trí tuệ nhân tạo": "TRÍ TUỆ NHÂN TẠO",
}

# Metadata column mapping: new file columns → standard columns
META_MAPPING: dict[str, str] = {
    "Mã SV": "Ma_SV",
    "Họ và tên": "Ho_Ten",
    "Ngày sinh": "Ngay_Sinh",
    "Lớp": "Lop",
}

# Columns to drop from new file (not needed in merged data)
DROP_COLUMNS: list[str] = ["STT"]


def load_new_data(filepath: str = NEW_DATA_PATH) -> pd.DataFrame:
    """Load and standardize the new CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"      ✅ Loaded new data: {len(df)} rows, {len(df.columns)} columns")
    print(f"      📋 Columns: {list(df.columns)}")

    # Drop unnecessary columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Rename metadata columns
    df = df.rename(columns=META_MAPPING)

    # Rename subject columns
    df = df.rename(columns=NEW_SUBJECT_MAPPING)

    print(f"      ✅ After rename: {list(df.columns)}")
    return df


def merge_data() -> pd.DataFrame:
    """Merge new data into existing FIT_HAU_Raw_Scores.csv."""
    print("=" * 60)
    print("🔄 DSS FIT-HAU — Merge New Data")
    print("=" * 60)

    # Step 1: Load existing data
    print("\n[1/4] Loading existing raw data...")
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {RAW_CSV_PATH}")

    df_existing = pd.read_csv(RAW_CSV_PATH, encoding='utf-8-sig')
    print(f"      ✅ Existing: {len(df_existing)} rows, {len(df_existing.columns)} columns")

    # Step 2: Load new data
    print("\n[2/4] Loading new data...")
    df_new = load_new_data()

    # Step 3: Add new subject columns to existing data if not present
    print("\n[3/4] Adding new subject columns...")
    new_subjects = [col for col in NEW_SUBJECT_MAPPING.values()]
    for subj in new_subjects:
        if subj not in df_existing.columns:
            df_existing[subj] = -1.0
            print(f"      ➕ Added column: {subj}")

    # Ensure all 24 subject columns exist in new data (fill missing with -1.0 = chưa học)
    all_subject_cols = [c for c in df_existing.columns if c not in ["Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"]]
    for col in all_subject_cols:
        if col not in df_new.columns:
            df_new[col] = -1.0

    # Step 4: Merge — update existing students or append new ones
    print("\n[4/4] Merging data...")

    existing_ids = set(df_existing["Ma_SV"].astype(str))
    updated_count = 0
    appended_count = 0

    for _, row in df_new.iterrows():
        ma_sv = str(row["Ma_SV"])
        if ma_sv in existing_ids:
            # Update: fill in the 4 new subject scores for existing students
            idx = df_existing[df_existing["Ma_SV"].astype(str) == ma_sv].index
            for subj in new_subjects:
                if subj in row.index and pd.notna(row[subj]):
                    df_existing.loc[idx, subj] = row[subj]
            updated_count += 1
        else:
            # Append new student row
            meta_cols_set = {"Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"}
            new_row = {}
            for col in df_existing.columns:
                if col in row.index:
                    new_row[col] = row[col]
                elif col in meta_cols_set:
                    new_row[col] = ""  # Metadata → chuỗi rỗng
                else:
                    new_row[col] = -1.0  # Score → chưa học
            df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            appended_count += 1

    print(f"      ✅ Updated existing students: {updated_count}")
    print(f"      ✅ Appended new students: {appended_count}")
    print(f"      ✅ Total students now: {len(df_existing)}")

    # Reorder columns: metadata first, then subjects alphabetically
    meta_cols = ["Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"]
    subj_cols = sorted([c for c in df_existing.columns if c not in meta_cols])
    df_existing = df_existing[meta_cols + subj_cols]

    # Save
    df_existing.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Saved → {RAW_CSV_PATH}")
    print(f"  📐 Shape: {df_existing.shape[0]} rows × {df_existing.shape[1]} columns")
    print("=" * 60)

    return df_existing


if __name__ == "__main__":
    merge_data()
