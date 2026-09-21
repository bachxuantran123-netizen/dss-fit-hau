"""
pdf_extractor.py — Trích xuất điểm thi từ file PDF bảng điểm HAU
=================================================================
Quét toàn bộ thư mục data/raw/ĐIỂM NĂM 24-25, trích xuất điểm TBCMH
của các môn thuộc ngành CNTT, gộp theo Mã SV thành 1 file CSV duy nhất.

Output: data/raw/FIT_HAU_Raw_Scores.csv
"""

import os
import sys
import re
import pdfplumber
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
RAW_BASE_DIR: str = os.path.join("data", "raw", "ĐIỂM NĂM 24-25")
OUTPUT_CSV: str = os.path.join("data", "raw", "FIT_HAU_Raw_Scores.csv")

# Mapping: keyword trong tên file PDF → tên cột chuẩn hóa trong CSV
# Key là chữ thường để so sánh linh hoạt
SUBJECT_MAPPING: dict[str, str] = {
    "an ninh mạng": "AN NINH MẠNG",
    "an toàn và bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    "công nghệ phần mềm": "CÔNG NGHỆ PHẦN MỀM",
    "hệ quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "cơ sở dữ liệu": "CƠ SỞ DỮ LIỆU",
    "gis và quản lý": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "hệ điều hành linux": "HỆ ĐIỀU HÀNH LINUX",
    "hệ điều hành": "HỆ ĐIỀU HÀNH",
    "kỹ thuật lập trình": "KỸ THUẬT LẬP TRÌNH",
    "kỹ thuật đồ hoạ": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật xử lý ảnh": "XỬ LÝ ẢNH",
    "lập trình hướng đối tượng": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "quản trị mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mạng": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "mạng máy tính": "MẠNG MÁY TÍNH",
    "nhập môn cntt": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "phân tích và thiết kế": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "toán rời rạc": "TOÁN RỜI RẠC",
    "xử lý tín hiệu": "XỬ LÝ TÍN HIỆU SỐ",
    "cấu trúc dữ liệu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "kiến trúc máy tính": "KIẾN TRÚC MÁY TÍNH",
}

# Column indices in PDF table (based on observed structure)
IDX_MA_SV: int = 1
IDX_HO_TEN: int = 2
IDX_NGAY_SINH: int = 3
IDX_LOP: int = 4
IDX_TBCMH_SO: int = 7  # Điểm TBCMH dạng số


# ============================================================
# FUNCTIONS
# ============================================================
def match_subject(filename: str) -> str | None:
    """Match a PDF filename to a standardized subject name.

    Uses longest-match-first strategy to avoid partial matches
    (e.g., 'hệ điều hành linux' must match before 'hệ điều hành').
    """
    fname_lower = filename.lower()
    # Sort by key length descending for longest match first
    for keyword in sorted(SUBJECT_MAPPING.keys(), key=len, reverse=True):
        if keyword in fname_lower:
            return SUBJECT_MAPPING[keyword]
    return None


def extract_scores_from_pdf(pdf_path: str) -> list[dict]:
    """Extract student scores from a single PDF file.

    Returns:
        list of dicts with keys: Ma_SV, Ho_Ten, Ngay_Sinh, Lop, Diem
    """
    records = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables:
                    continue
                for table in tables:
                    for row in table:
                        if row is None or len(row) < 8:
                            continue
                        ma_sv = row[IDX_MA_SV]
                        if ma_sv is None or not re.match(r'^\d{7,}', str(ma_sv).strip()):
                            continue  # Skip header/non-student rows

                        diem_str = row[IDX_TBCMH_SO]
                        try:
                            diem = float(str(diem_str).replace(',', '.').strip())
                        except (ValueError, TypeError):
                            diem = 0.0

                        records.append({
                            "Ma_SV": str(ma_sv).strip(),
                            "Ho_Ten": str(row[IDX_HO_TEN] or "").strip(),
                            "Ngay_Sinh": str(row[IDX_NGAY_SINH] or "").strip(),
                            "Lop": str(row[IDX_LOP] or "").strip(),
                            "Diem": diem,
                        })
    except Exception as e:
        print(f"      ⚠️  Lỗi đọc PDF {pdf_path}: {e}")
    return records


def scan_all_pdfs() -> pd.DataFrame:
    """Scan all PDF files, extract IT subject scores, and merge by Ma_SV.

    Returns:
        pd.DataFrame with columns: Ma_SV, Ho_Ten, Ngay_Sinh, Lop,
        <subject_1>, <subject_2>, ..., <subject_N>
    """
    all_records: dict[str, dict] = {}  # Ma_SV -> {subject: score, ...}
    subject_set: set[str] = set()
    files_processed = 0
    files_skipped = 0

    for semester_dir in sorted(os.listdir(RAW_BASE_DIR)):
        semester_path = os.path.join(RAW_BASE_DIR, semester_dir)
        if not os.path.isdir(semester_path):
            continue

        print(f"\n  📁 Scanning: {semester_dir}")
        for pdf_file in sorted(os.listdir(semester_path)):
            if not pdf_file.endswith(".pdf"):
                continue

            subject_name = match_subject(pdf_file)
            if subject_name is None:
                files_skipped += 1
                continue

            pdf_path = os.path.join(semester_path, pdf_file)
            records = extract_scores_from_pdf(pdf_path)
            files_processed += 1
            subject_set.add(subject_name)

            print(f"      ✅ {pdf_file} → {subject_name} ({len(records)} SV)")

            for rec in records:
                ma_sv = rec["Ma_SV"]
                if ma_sv not in all_records:
                    all_records[ma_sv] = {
                        "Ma_SV": ma_sv,
                        "Ho_Ten": rec["Ho_Ten"],
                        "Ngay_Sinh": rec["Ngay_Sinh"],
                        "Lop": rec["Lop"],
                    }
                # Ghi điểm vào cột tương ứng — nếu trùng thì giữ điểm cao nhất
                current = all_records[ma_sv].get(subject_name, 0.0)
                all_records[ma_sv][subject_name] = max(current, rec["Diem"])

    print(f"\n  📊 Summary:")
    print(f"      Files processed: {files_processed}")
    print(f"      Files skipped (non-IT): {files_skipped}")
    print(f"      Unique students: {len(all_records)}")
    print(f"      Subjects found: {len(subject_set)}")
    print(f"      Subjects: {sorted(subject_set)}")

    # Build DataFrame
    df = pd.DataFrame(list(all_records.values()))

    # Fill missing subjects with -1.0 (to distinguish from a real score of 0.0)
    for subj in sorted(subject_set):
        if subj not in df.columns:
            df[subj] = -1.0
    df = df.fillna(-1.0)

    # Reorder columns: metadata first, then subjects alphabetically
    meta_cols = ["Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"]
    subj_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + subj_cols]

    return df


def run_extraction() -> pd.DataFrame:
    """Run the full PDF extraction pipeline."""
    print("=" * 60)
    print("📄 DSS FIT-HAU — PDF Score Extractor")
    print("=" * 60)

    if not os.path.exists(RAW_BASE_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {RAW_BASE_DIR}")

    df = scan_all_pdfs()

    # Guard: đảm bảo đã trích xuất được dữ liệu
    if df.empty:
        raise ValueError(
            "Không trích xuất được dữ liệu nào từ PDF. "
            "Kiểm tra lại thư mục PDF và SUBJECT_MAPPING có khớp không."
        )

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  💾 Saved → {OUTPUT_CSV}")
    print(f"  📐 Shape: {df.shape[0]} students × {df.shape[1]} columns")
    print("=" * 60)

    return df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_extraction()
