"""
data_pipeline.py — Sprint 1: Data Preprocessing & Labeling
===========================================================
Module xử lý dữ liệu thô từ Kaggle (Student Performance Dataset)
và chuyển đổi sang định dạng FIT-HAU phục vụ huấn luyện Decision Tree.

Pipeline:
    1. Load raw CSV (student-mat.csv)
    2. Xử lý Missing Values
    3. Chọn lọc & đổi tên Feature (G1, G2, G3 → Toán Rời Rạc, Lập Trình C, CSDL)
    4. Quy đổi thang điểm 20 → thang 10
    5. Gán nhãn chuyên ngành (AI, SE, Cảnh báo)
    6. Xuất file CSV sạch → data/processed/FIT_HAU_Cleaned.csv
"""

import os
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
RAW_DATA_PATH: str = os.path.join("data", "raw", "student-mat.csv")
PROCESSED_DATA_PATH: str = os.path.join("data", "processed", "FIT_HAU_Cleaned.csv")

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0
KAGGLE_MAX_SCORE: float = 20.0

# Mapping Kaggle columns → FIT-HAU column names
COLUMN_MAPPING: dict[str, str] = {
    "G1": "Toan_Roi_Rac",
    "G2": "Lap_Trinh_C",
    "G3": "Co_So_Du_Lieu",
}

# Features used for training
FEATURE_COLUMNS: list[str] = list(COLUMN_MAPPING.values())
TARGET_COLUMN: str = "Chuyen_Nganh"


# ============================================================
# FUNCTIONS
# ============================================================
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset from Kaggle."""
    # TODO: Sprint 1 - Task 1
    raise NotImplementedError("Sprint 1 - Task 1: Implement data loading")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and report dropped/filled rows."""
    # TODO: Sprint 1 - Task 2
    raise NotImplementedError("Sprint 1 - Task 2: Implement missing value handling")


def rename_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns (G1, G2, G3) and rename to FIT-HAU names."""
    # TODO: Sprint 1 - Task 3
    raise NotImplementedError("Sprint 1 - Task 3: Implement feature mapping")


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Convert scores from Kaggle scale (0-20) to FIT-HAU scale (0-10).

    Formula: score_10 = score_20 * (10 / 20)
    Uses Pandas vectorization — NO for-loops allowed.
    """
    # TODO: Sprint 1 - Task 4 (part 1)
    raise NotImplementedError("Sprint 1 - Task 4a: Implement score normalization")


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign major labels based on normalized scores using hard-coded if-else rules.

    Labels:
        - 'AI'       : Strong in math-heavy subjects
        - 'SE'       : Strong in programming subjects
        - 'Cảnh báo' : Below threshold — academic warning
    """
    # TODO: Sprint 1 - Task 4 (part 2)
    raise NotImplementedError("Sprint 1 - Task 4b: Implement labeling rules")


def run_pipeline() -> pd.DataFrame:
    """Execute the full data pipeline end-to-end.

    Returns:
        pd.DataFrame: Cleaned and labeled dataset ready for training.
    """
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
    print(f"      ✅ Rows: {rows_before} → {len(df)} (removed {rows_before - len(df)})")

    # Step 3: Feature mapping
    print("\n[3/5] Renaming & selecting features...")
    df = rename_and_select_features(df)
    print(f"      ✅ Columns: {list(df.columns)}")

    # Step 4: Normalize
    print("\n[4/5] Normalizing scores (20 → 10)...")
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
