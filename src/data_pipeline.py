"""
data_pipeline.py — Data Preprocessing & Labeling (v3 — Content-Based Filtering)
=================================================================================
Module xử lý dữ liệu điểm sinh viên FIT-HAU từ file CSV đã trích xuất từ PDF.

KIẾN TRÚC HYBRID 2 TẦNG:
    Tầng 1 (file này): Content-Based Filtering
        → Gán nhãn 5 hướng nghề nghiệp bằng Cosine Similarity
          giữa vector điểm sinh viên và Career Profile Vectors.
    Tầng 2 (train_core.py): Decision Tree
        → Học từ nhãn đã gán, dự đoán và giải thích cho user mới (XAI).

Pipeline: FIT_HAU_Raw_Scores.csv → Clean → Label (Cosine Sim) → FIT_HAU_Cleaned.csv
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
# FEATURE ORDER — Thứ tự chuẩn 20 môn (alphabet tiếng Việt)
# ============================================================
# Thứ tự này PHẢI khớp chính xác với thứ tự các giá trị trong
# CAREER_PROFILES bên dưới. Khi thêm/bớt môn, cập nhật CẢ HAI.
FEATURE_ORDER: list[str] = [
    "AN NINH MẠNG",                      # 0
    "AN TOÀN VÀ BẢO MẬT HTTT",           # 1
    "C#",                                 # 2  [NEW]
    "CÔNG NGHỆ PHẦN MỀM",                # 3
    "CƠ SỞ DỮ LIỆU",                    # 4
    "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",   # 5
    "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH", # 6
    "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",       # 7
    "HỆ ĐIỀU HÀNH",                      # 8
    "HỆ ĐIỀU HÀNH LINUX",                # 9
    "JAVA",                               # 10 [NEW]
    "KIẾN TRÚC MÁY TÍNH",                # 11
    "KỸ THUẬT LẬP TRÌNH",                # 12
    "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",         # 13
    "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",        # 14
    "LẬP TRÌNH WEB",                      # 15 [NEW]
    "MẠNG MÁY TÍNH",                     # 16
    "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",    # 17
    "PHÂN TÍCH VÀ THIẾT KẾ HTTT",        # 18
    "QUẢN TRỊ MẠNG MÁY TÍNH",           # 19
    "TOÁN RỜI RẠC",                      # 20
    "TRÍ TUỆ NHÂN TẠO",                  # 21 [NEW]
    "XỬ LÝ TÍN HIỆU SỐ",               # 22
    "XỬ LÝ ẢNH",                         # 23
]

# ============================================================
# CAREER PROFILE VECTORS — Content-Based Filtering
# ============================================================
# Mỗi Career Profile là 1 vector 20 chiều, giá trị [0.0, 1.0]:
#   0.0 = môn không liên quan đến ngành này
#   1.0 = môn cốt lõi (core competency) của ngành này
#
# Cosine Similarity so sánh HƯỚNG (pattern) điểm, không phụ thuộc
# tổng điểm tuyệt đối → sinh viên giỏi đều vẫn được phân biệt
# nếu pattern điểm nghiêng về 1 nhóm.
#
# Thứ tự giá trị trong mỗi vector PHẢI khớp với FEATURE_ORDER.
# ============================================================
CAREER_PROFILES: dict[str, np.ndarray] = {
    # Thứ tự 24 features (alphabet):
    # ANM, ATBM, C#, CNPM, CSDL, CTDL, GIS, HQTCSDL, HDH, HDHL,
    # JAVA, KTMT, KTLT, KTDHMT, LTHDTG, LTWEB, MMT, NMCNTT, PTTKHTTT, QTMMT,
    # TRR, TTNT, XLTHS, XLA
    "Software Engineer": np.array([
        # ANM  ATBM  C#    CNPM  CSDL  CTDL  GIS   HQTCSDL  HDH  HDHL
        0.1,  0.1,  0.8,  1.0,  0.5,  0.9,  0.2,  0.4,     0.3, 0.2,
        # JAVA KTMT  KTLT  KTDHMT  LTHDTG  LTWEB  MMT  NMCNTT  PTTKHTTT  QTMMT
        0.9,  0.3,  0.9,  0.3,    0.9,    0.7,   0.2, 0.3,    0.8,      0.1,
        # TRR  TTNT  XLTHS  XLA
        0.5,  0.3,  0.2,   0.2,
    ]),
    "Data Engineer": np.array([
        0.1,  0.2,  0.5,  0.4,  1.0,  0.5,  0.5,  0.9,     0.2, 0.3,
        0.4,  0.2,  0.4,  0.2,    0.4,    0.3,   0.2, 0.3,    0.6,      0.2,
        0.4,  0.5,  0.3,   0.2,
    ]),
    "AI Engineer": np.array([
        0.0,  0.0,  0.3,  0.3,  0.3,  0.8,  0.4,  0.2,     0.1, 0.1,
        0.4,  0.2,  0.5,  0.7,    0.5,    0.2,   0.1, 0.3,    0.3,      0.1,
<<<<<<< HEAD
        0.9,  1.0,  0.8,   0.9,
=======
        0.9,  1.0,  0.3,   0.9,
>>>>>>> fc264467222021095a802c504e6090b811d0e1a8
    ]),
    "Security Engineer": np.array([
        1.0,  0.9,  0.2,  0.2,  0.2,  0.2,  0.1,  0.2,     0.5, 0.5,
        0.2,  0.4,  0.3,  0.1,    0.2,    0.2,   0.9, 0.3,    0.2,      0.8,
        0.2,  0.1,  0.1,   0.1,
    ]),
    "System/DevOps": np.array([
        0.3,  0.3,  0.3,  0.3,  0.2,  0.2,  0.2,  0.3,     0.9, 1.0,
        0.3,  0.8,  0.3,  0.1,    0.2,    0.3,   0.7, 0.3,    0.2,      0.6,
        0.1,  0.1,  0.1,   0.1,
    ]),
}


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
    """Clean data: ensure numeric scores, fill NaN with -1.0 (chưa học)."""
    df_clean = df.copy()

    # Identify score columns (all except metadata)
    score_cols = [c for c in df_clean.columns if c not in NON_FEATURE_COLUMNS]

    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(-1.0).round(1)

    return df_clean


def cosine_similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính Cosine Similarity giữa 2 vectors.

    Cosine Similarity = (A · B) / (||A|| × ||B||)

    Returns:
        float trong khoảng [-1.0, 1.0], thường [0.0, 1.0] vì điểm >= 0.
        Trả về 0.0 nếu một trong hai vector có norm = 0 (toàn số 0).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Gán nhãn chuyên ngành bằng Content-Based Filtering (Cosine Similarity).

    Logic:
        1. Lấy vector điểm 20 môn của sinh viên (theo FEATURE_ORDER).
        2. Tính Cosine Similarity với từng Career Profile Vector.
        3. Gán nhãn = ngành có similarity cao nhất.
        4. Nếu tất cả similarity = 0 (toàn điểm 0) → gán "Software Engineer".

    So với SKILL_MATRIX cũ:
        - SKILL_MATRIX: chỉ xét nhóm con môn học, so trung bình → thiên lệch
        - Cosine Sim: xét TOÀN BỘ 20 môn đồng thời, đo pattern → công bằng hơn
    """
    df_labeled = df.copy()
    labels = []
    similarities_log = []  # Để in thống kê

    # Xác định các feature thực tế có trong DataFrame
    valid_features = [f for f in FEATURE_ORDER if f in df_labeled.columns]
    missing_features = [f for f in FEATURE_ORDER if f not in df_labeled.columns]

    if missing_features:
        print(f"      ⚠️  Missing features (sẽ dùng giá trị 0): {missing_features}")

    for _, row in df_labeled.iterrows():
        # Xây dựng student vector theo FEATURE_ORDER, loại bỏ điểm -1.0 để tính Cosine Similarity (tránh góc âm)
        student_vector = np.array([
            max(0.0, float(row[f])) if f in df_labeled.columns else 0.0
            for f in FEATURE_ORDER
        ])

        # Tính similarity với từng Career Profile
        best_career = "Software Engineer"  # Fallback
        best_sim = -1.0
        sim_dict = {}

        for career, profile_vector in CAREER_PROFILES.items():
            sim = cosine_similarity_score(student_vector, profile_vector)
            sim_dict[career] = sim
            if sim > best_sim:
                best_sim = sim
                best_career = career

        labels.append(best_career)
        similarities_log.append(sim_dict)

    df_labeled[TARGET_COLUMN] = labels

    # In thống kê similarity trung bình theo ngành
    if similarities_log:
        sim_df = pd.DataFrame(similarities_log)
        print("      📊 Cosine Similarity trung bình theo Career Profile:")
        for career in CAREER_PROFILES.keys():
            mean_sim = sim_df[career].mean()
            print(f"         {career}: {mean_sim:.4f}")

    return df_labeled


def prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Remove metadata columns, keep only score features + target label."""
    cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df_final = df.drop(columns=cols_to_drop)
    return df_final


def run_pipeline() -> pd.DataFrame:
    """Execute the full data pipeline end-to-end."""
    print("=" * 60)
    print("🚀 DSS FIT-HAU — Data Pipeline v3 (Content-Based Filtering)")
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

    # Step 3: Label (Content-Based Filtering)
    print("\n[3/4] Assigning career labels (Cosine Similarity)...")
    df = assign_labels(df)
    print(f"\n      ✅ Label distribution:")
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