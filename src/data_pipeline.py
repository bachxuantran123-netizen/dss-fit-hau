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
# Mỗi Career Profile là 1 vector 24 chiều, giá trị [0.0, 1.0]:
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
        0.9,  1.0,  0.3,   0.9,
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

    # Pandas vectorization (No for loop)
    df_clean[score_cols] = df_clean[score_cols].apply(pd.to_numeric, errors='coerce').fillna(-1.0).round(1)

    # Khắc phục lỗi dữ liệu cũ: thay thế 0.0 thành -1.0 (coi như chưa học)
    df_clean[score_cols] = df_clean[score_cols].replace(0.0, -1.0)

    return df_clean


def cosine_similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính Cosine Similarity giữa vector điểm sinh viên và vector nghề nghiệp.

    Logic:
        - Sinh viên học môn nào thì chỉ cắt lấy các môn đó ra để tính Cosine Similarity.
        - Môn nào chưa học (điểm == -1.0 hoặc < 0) thì bỏ qua hoàn toàn, không tính vào vector.
        - Cả vector sinh viên (vec_a) và vector nghề nghiệp (vec_b) đều được cắt theo các môn đã học.

    Cosine Similarity = (A_sub · B_sub) / (||A_sub|| × ||B_sub||)

    Returns:
        float trong khoảng [0.0, 1.0].
        Trả về 0.0 nếu sinh viên chưa học môn nào hoặc norm = 0.
    """
    valid_mask = (vec_a != -1.0) & (vec_a >= 0)
    if not np.any(valid_mask):
        return 0.0

    sub_a = vec_a[valid_mask]
    sub_b = vec_b[valid_mask]

    norm_a = np.linalg.norm(sub_a)
    norm_b = np.linalg.norm(sub_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(sub_a, sub_b) / (norm_a * norm_b))


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Gán nhãn chuyên ngành bằng Content-Based Filtering (Cosine Similarity).

    Cải tiến (Dynamic Slicing theo môn đã học):
        - Thay vì so sánh cả 24 môn (coi môn chưa học là 0 làm sai lệch chuẩn vector nghề nghiệp),
          sinh viên học môn nào thì chỉ cắt lấy các môn đó ra để tính Cosine Similarity.
        - Môn nào chưa học (-1.0 hoặc < 0) thì bỏ qua hoàn toàn, không tính vào vector.
        - Vector nghề nghiệp cũng chỉ cắt lấy đúng các môn tương ứng mà sinh viên đã học.

    Công thức:
        Với mỗi sinh viên i có tập môn đã học K_i:
        - Vector sinh viên: s_{K_i} = [score_{i, j}]_{j in K_i}
        - Vector nghề nghiệp c: p_{c, K_i} = [profile_{c, j}]_{j in K_i}
        - Cosine Similarity = (s_{K_i} · p_{c, K_i}) / (||s_{K_i}|| × ||p_{c, K_i}||)
    """
    df_labeled = df.copy()

    # Xác định các feature thực tế có trong DataFrame
    missing_features = [f for f in FEATURE_ORDER if f not in df_labeled.columns]
    if missing_features:
        print(f"      ⚠️  Missing features (sẽ dùng giá trị -1.0): {missing_features}")

    # 1. Trích xuất ma trận điểm theo FEATURE_ORDER, điền -1.0 cho các môn thiếu (chưa học)
    score_matrix = df_labeled.reindex(columns=FEATURE_ORDER, fill_value=-1.0).values

    # Xác định mask các môn đã học (khác -1.0 và >= 0)
    # Môn nào chưa học (-1.0) thì bỏ qua hoàn toàn khỏi vector
    valid_mask = (score_matrix != -1.0) & (score_matrix >= 0)
    calc_matrix = np.where(valid_mask, score_matrix, 0.0)

    # 2. Xây dựng ma trận Profile (M ngành × 24 môn)
    careers = list(CAREER_PROFILES.keys())
    profile_matrix = np.array([CAREER_PROFILES[c] for c in careers])

    # 3. Tính Cosine Similarity chỉ trên các môn sinh viên đã học:
    # Tử số: Tích vô hướng (A · B) chỉ trên các môn đã học (calc_matrix = 0 ở môn chưa học)
    dot_products = np.dot(calc_matrix, profile_matrix.T)  # Shape (N, M)

    # Chuẩn của vector sinh viên ||A_sub|| (chỉ tính trên các môn đã học)
    student_norms = np.linalg.norm(calc_matrix, axis=1)  # Shape (N,)

    # Chuẩn của vector nghề nghiệp ||B_sub|| CŨNG CHỈ TÍNH TRÊN CÁC MÔN SINH VIÊN ĐÃ HỌC
    # Môn chưa học bị loại bỏ hoàn toàn khỏi chuẩn độ dài của profile
    profile_sq = profile_matrix ** 2
    sub_profile_norm_sq = np.dot(valid_mask.astype(float), profile_sq.T)
    sub_profile_norms = np.sqrt(sub_profile_norm_sq)  # Shape (N, M)

    # Mẫu số: ||A_sub|| × ||B_sub||
    denominator = student_norms[:, np.newaxis] * sub_profile_norms  # Shape (N, M)

    # Tính Cosine Similarity, tránh chia cho 0 khi mẫu số <= 0 hoặc SV chưa học môn nào
    similarity_matrix = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products),
        where=denominator > 1e-9
    )

    # Lấy nhãn cao nhất
    best_career_indices = np.argmax(similarity_matrix, axis=1)
    labels = np.array([careers[i] for i in best_career_indices])

    # Fallback cho trường hợp toàn 0 (chưa học môn nào hoặc tất cả similarity = 0)
    zero_norm_mask = (student_norms == 0) | (np.max(similarity_matrix, axis=1) == 0)
    labels[zero_norm_mask] = "Software Engineer"

    df_labeled[TARGET_COLUMN] = labels

    # In thống kê similarity trung bình theo ngành (chỉ tính trên các sinh viên có điểm)
    active_mask = ~zero_norm_mask
    sim_df = pd.DataFrame(similarity_matrix, columns=careers)
    print("      📊 Cosine Similarity trung bình theo Career Profile (trên các môn đã học):")
    for career in careers:
        avg_sim = sim_df.loc[active_mask, career].mean() if active_mask.any() else 0.0
        print(f"         {career}: {avg_sim:.4f}")

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