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
import difflib
import unicodedata
import pdfplumber
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONSTANTS
# ============================================================
RAW_BASE_DIR: str = os.path.join("data", "raw", "ĐIỂM NĂM 24-25")
OUTPUT_CSV: str = os.path.join("data", "raw", "FIT_HAU_Raw_Scores.csv")

STANDARD_SUBJECTS: list[str] = [
    "AN NINH MẠNG",
    "AN TOÀN VÀ BẢO MẬT HTTT",
    "C#",
    "CÔNG NGHỆ PHẦN MỀM",
    "CƠ SỞ DỮ LIỆU",
    "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "HỆ ĐIỀU HÀNH",
    "HỆ ĐIỀU HÀNH LINUX",
    "JAVA",
    "KIẾN TRÚC MÁY TÍNH",
    "KỸ THUẬT LẬP TRÌNH",
    "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "LẬP TRÌNH WEB",
    "MẠNG MÁY TÍNH",
    "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "QUẢN TRỊ MẠNG MÁY TÍNH",
    "TOÁN RỜI RẠC",
    "TRÍ TUỆ NHÂN TẠO",
    "XỬ LÝ TÍN HIỆU SỐ",
    "XỬ LÝ ẢNH",
]

# Mã học phần HAU -> Môn chuẩn (24 môn)
SUBJECT_CODES: dict[str, str] = {
    "TH5219": "AN NINH MẠNG",
    "TH5210": "AN TOÀN VÀ BẢO MẬT HTTT",
    "TH4315": "C#",
    "TH4306": "CÔNG NGHỆ PHẦN MỀM",
    "TH5217": "CƠ SỞ DỮ LIỆU",
    "TH4303": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "DT1926": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "TH5221": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "TH5203": "HỆ ĐIỀU HÀNH",
    "TH5211": "HỆ ĐIỀU HÀNH LINUX",
    "TH4316": "JAVA",
    "TH4319": "KIẾN TRÚC MÁY TÍNH",
    "TH4304": "KỸ THUẬT LẬP TRÌNH",
    "TH5231": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "TH4305": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "TH4309": "LẬP TRÌNH WEB",
    "TH5206": "MẠNG MÁY TÍNH",
    "TH5201": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "TH5208": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "TH5218": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "TH4302": "TOÁN RỜI RẠC",
    "TH4320": "TRÍ TUỆ NHÂN TẠO",
    "TH5205": "XỬ LÝ TÍN HIỆU SỐ",
    "TH5209": "XỬ LÝ ẢNH",
}

# Tên môn / Từ khóa / Tên viết tắt -> Môn chuẩn
_ALIASES_RAW: dict[str, str] = {
    # An ninh mạng
    "an ninh mạng": "AN NINH MẠNG",
    "an ninh mang": "AN NINH MẠNG",
    # ATBM
    "an toàn và bảo mật hệ thống thông tin": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn và bảo mật httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn bảo mật httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn và bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    # C#
    "ngôn ngữ c# và công nghệ .net": "C#",
    "ngôn ngữ c#": "C#",
    "lập trình c#": "C#",
    "c#": "C#",
    # CNPM
    "công nghệ phần mềm": "CÔNG NGHỆ PHẦN MỀM",
    "cnpm": "CÔNG NGHỆ PHẦN MỀM",
    # CSDL
    "cơ sở dữ liệu": "CƠ SỞ DỮ LIỆU",
    "csdl": "CƠ SỞ DỮ LIỆU",
    # CTDL&GT
    "cấu trúc dữ liệu và giải thuật": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "cấu trúc dữ liệu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "ctdl": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    # GIS
    "gis và quản lý đô thị thông minh": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "gis và quản lý đô thị": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "gis và quản lý": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "gis": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    # HQTCSDL
    "hệ quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị csdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hqtcsdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    # HĐH
    "hệ điều hành linux": "HỆ ĐIỀU HÀNH LINUX",
    "hệ điều hành": "HỆ ĐIỀU HÀNH",
    # Java
    "công nghệ java": "JAVA",
    "lập trình java": "JAVA",
    "java": "JAVA",
    # KTMT
    "kiến trúc máy tính": "KIẾN TRÚC MÁY TÍNH",
    "ktmt": "KIẾN TRÚC MÁY TÍNH",
    # KTLT
    "kỹ thuật lập trình": "KỸ THUẬT LẬP TRÌNH",
    "ktlt": "KỸ THUẬT LẬP TRÌNH",
    # Đồ hoạ
    "kỹ thuật đồ họa máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ hoạ máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "đồ họa máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "đồ hoạ máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ họa": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ hoạ": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    # OOP
    "lập trình hướng đối tượng": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "hướng đối tượng": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "oop": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "lthdt": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    # Web
    "công nghệ web": "LẬP TRÌNH WEB",
    "lập trình web": "LẬP TRÌNH WEB",
    # Mạng máy tính
    "mạng máy tính": "MẠNG MÁY TÍNH",
    # Nhập môn
    "nhập môn cntt và truyền thông": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhập môn cntt": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhập môn công nghệ thông tin": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    # PTTKHT
    "phân tích và thiết kế hệ thống thông tin": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích và thiết kế httt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích thiết kế hệ thống": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích và thiết kế": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "pttkhttt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    # QTMMT
    "quản trị mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mạng": "QUẢN TRỊ MẠNG MÁY TÍNH",
    # Toán rời rạc
    "toán rời rạc": "TOÁN RỜI RẠC",
    # TTNT
    "trí tuệ nhân tạo": "TRÍ TUỆ NHÂN TẠO",
    "ttnt": "TRÍ TUỆ NHÂN TẠO",
    # XLTHS
    "xử lý tín hiệu số": "XỬ LÝ TÍN HIỆU SỐ",
    "xử lý tín hiệu": "XỬ LÝ TÍN HIỆU SỐ",
    "xlths": "XỬ LÝ TÍN HIỆU SỐ",
    # Xử lý ảnh
    "xử lý ảnh": "XỬ LÝ ẢNH",
    "kỹ thuật xử lý ảnh": "XỬ LÝ ẢNH",
}

# Tương thích ngược: SUBJECT_MAPPING tham chiếu _ALIASES_RAW
SUBJECT_MAPPING: dict[str, str] = _ALIASES_RAW

# Column indices in PDF table (based on observed structure)
IDX_MA_SV: int = 1
IDX_HO_TEN: int = 2
IDX_NGAY_SINH: int = 3
IDX_LOP: int = 4
IDX_TBCMH_SO: int = 7  # Điểm TBCMH dạng số

SCORE_PATTERN = re.compile(r"(\d{1,2}[.,]\d+|\d{1,2})\s*$")

def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt (O(n), xử lý cả đ/Đ)."""
    s = unicodedata.normalize("NFD", input_str)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    """NFC, lowercase, bỏ ký tự phân cách, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[_\-–—|/\\]", " ", text)
    return re.sub(r"\s+", " ", text)


def _norm_key(text: str) -> str:
    """Khoá so khớp: không dấu, chữ thường, chỉ giữ [a-z0-9#] và khoảng trắng."""
    s = remove_accents(unicodedata.normalize("NFC", text).lower())
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ALIASES: dict[str, str] = {_norm_key(k): v for k, v in _ALIASES_RAW.items()}
_ALIASES_SORTED: list[str] = sorted(_ALIASES, key=len, reverse=True)
_FUZZY_KEYS: list[str] = [k for k in _ALIASES if len(k) >= 6]


# ============================================================
# FUNCTIONS
# ============================================================
def match_subject(text: str) -> str | None:
    """Map Mã học phần hoặc Tên môn sang 24 môn chuẩn.

    Hỗ trợ:
    1) So khớp Mã học phần (vd: TH5219, TH4315, DT1926.1...)
    2) So khớp chính xác hoặc chuỗi con (không dấu, xử lý lệch dấu tiếng Việt)
    3) So khớp mờ (difflib) cho trường hợp sai sót chính tả / ngắt từ nhẹ.
    """
    if not text:
        return None
    raw_str = str(text).strip()
    if not raw_str:
        return None

    # 1. Match theo mã học phần (vd TH5209, TH4315, DT1926...)
    for code, subj in SUBJECT_CODES.items():
        if re.search(rf"\b{re.escape(code)}(?:\.\d+)?\b", raw_str, re.I):
            return subj

    # 2. Khớp theo alias tên môn (bỏ dấu tiếng Việt, chống lệch quy tắc đặt dấu)
    n = _norm_key(raw_str)
    if not n:
        return None
    if n in _ALIASES:
        return _ALIASES[n]
    for alias in _ALIASES_SORTED:
        if re.search(rf"(?<![a-z0-9#]){re.escape(alias)}(?![a-z0-9#])", n):
            return _ALIASES[alias]

    # 3. Fuzzy match (cho tên môn dài >= 6 ký tự để bắt sai sót nhẹ)
    if len(n) >= 6:
        close = difflib.get_close_matches(n, _FUZZY_KEYS, n=1, cutoff=0.82)
        if close:
            return _ALIASES[close[0]]

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
    if not all_records:
        return pd.DataFrame(columns=["Ma_SV", "Ho_Ten", "Ngay_Sinh", "Lop"])
        
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
# PDF READER (cho bảng điểm định dạng PDF của 1 sinh viên)
# ============================================================
def extract_scores_from_student_pdf(pdf_file_or_bytes, save_csv_path=None) -> dict:
    """Trích xuất điểm từ bảng điểm cá nhân dạng PDF.
    
    Đọc toàn bộ file PDF, trích xuất bảng, xuất ra CSV nếu có save_csv_path.
    Quét qua các dòng để tìm tên môn và điểm TBCMH.
    """
    scores = {}
    matched_lines = []
    unmatched_lines = []
    raw_texts = []
    warnings = []
    
    try:
        all_rows = []
        with pdfplumber.open(pdf_file_or_bytes) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                # Fallback 1: nếu PDF không có viền bảng (grid lines)
                if not tables:
                    tables = page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
                # Fallback 2: thử với tolerance cao hơn
                if not tables:
                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                    })
                    
                for table in tables:
                    for row in table:
                        if row is not None:
                            cleaned_row = [str(cell).strip().replace('\n', ' ') if cell else "" for cell in row]
                            all_rows.append(cleaned_row)
                            raw_texts.append(" | ".join(cleaned_row))
                
                # Fallback 3: Nếu vẫn không tìm được bảng, trích xuất text thuần
                # và tách thành các dòng riêng biệt
                if not tables:
                    page_text = page.extract_text()
                    if page_text:
                        for line in page_text.split('\n'):
                            line = line.strip()
                            if line:
                                cleaned_row = [line]
                                all_rows.append(cleaned_row)
                                raw_texts.append(line)
                            
        # Xuất file CSV theo yêu cầu
        if save_csv_path and all_rows:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df = pd.DataFrame(all_rows)
            df.to_csv(save_csv_path, index=False, header=False, encoding='utf-8-sig')
            
        # Khởi tạo vị trí cột
        subject_col_idx = -1
        score_col_idx = -1
        code_col_idx = -1
        
        pending_subj = ""
        pending_score = ""
        pending_raw = ""
        
        # Tìm điểm từ các dòng
        for row in all_rows:
            lower_row = [str(c).lower().strip() for c in row]
            
            # Kiểm tra xem dòng này có phải là dòng tiêu đề (header) không
            temp_subj_idx = -1
            temp_score_idx = -1
            temp_code_idx = -1
            
            for i, cell_val in enumerate(lower_row):
                if any(k in cell_val for k in ["ký hiệu", "mã học phần", "mã hp", "mã mh"]):
                    temp_code_idx = i
                if any(k in cell_val for k in ["tên học phần", "tên môn", "tên môn học"]):
                    temp_subj_idx = i
                elif temp_subj_idx == -1 and any(k in cell_val for k in ["học phần", "môn học"]) and "mã" not in cell_val and "ký hiệu" not in cell_val:
                    temp_subj_idx = i
                if any(k in cell_val for k in ["tbchp", "tbcmh", "điểm hệ 10", "tổng kết", "điểm tk", "thang 10", "tbcht", "điểm hp (10)"]):
                    temp_score_idx = i
                    
            if temp_subj_idx != -1 or temp_score_idx != -1:
                # Cập nhật vị trí cột cho bảng hiện tại
                if temp_subj_idx != -1:
                    subject_col_idx = temp_subj_idx
                # Lưu ý quan trọng: Nếu bảng không có cột điểm (ví dụ bảng "Các học phần chưa tích lũy"),
                # đặt score_col_idx = -1 để không nhận diện nhầm số tiết / tín chỉ thành điểm số!
                score_col_idx = temp_score_idx
                if temp_code_idx != -1:
                    code_col_idx = temp_code_idx
                pending_subj = ""
                pending_score = ""
                pending_raw = ""
                continue
                
            # Nếu bảng hiện tại không có cột điểm (ví dụ bảng "Các học phần chưa tích lũy"), bỏ qua dòng này
            if score_col_idx == -1 and temp_score_idx == -1:
                if subject_col_idx != -1:
                    continue

            # Xử lý đặc biệt cho dòng text thuần (1 cột duy nhất, từ fallback 3)
            # Quét toàn bộ dòng để tìm cả tên môn lẫn điểm
            if len(row) == 1 and subject_col_idx == -1:
                full_line = str(row[0]).strip()
                matched = match_subject(full_line)
                if matched:
                    # Tìm điểm trong dòng
                    score_match = SCORE_PATTERN.search(full_line)
                    if score_match:
                        score_str = score_match.group(1).replace(',', '.')
                        try:
                            score_val = float(score_str)
                            if 0.0 <= score_val <= 10.0:
                                scores[matched] = max(scores.get(matched, 0.0), score_val)
                                matched_lines.append({
                                    "subject": matched,
                                    "score": score_val,
                                    "ocr_text": full_line,
                                    "confidence": 1.0
                                })
                                continue
                        except (ValueError, TypeError):
                            pass
                    # Match môn nhưng chưa có điểm → pending
                    pending_subj = full_line
                    pending_score = ""
                    pending_raw = full_line
                    continue
                elif pending_subj:
                    # Dòng không match môn nhưng có pending → thử nối
                    combined = (pending_subj + " " + full_line).strip()
                    matched_combined = match_subject(combined)
                    score_match = SCORE_PATTERN.search(full_line)
                    if matched_combined and score_match:
                        score_str = score_match.group(1).replace(',', '.')
                        try:
                            score_val = float(score_str)
                            if 0.0 <= score_val <= 10.0:
                                scores[matched_combined] = max(scores.get(matched_combined, 0.0), score_val)
                                matched_lines.append({
                                    "subject": matched_combined,
                                    "score": score_val,
                                    "ocr_text": combined,
                                    "confidence": 1.0
                                })
                                pending_subj = ""
                                pending_score = ""
                                pending_raw = ""
                                continue
                        except (ValueError, TypeError):
                            pass
                    # Dòng chứa điểm cho pending subject?
                    if not matched_combined and pending_subj:
                        pend_matched = match_subject(pending_subj)
                        if pend_matched and score_match:
                            score_str = score_match.group(1).replace(',', '.')
                            try:
                                score_val = float(score_str)
                                if 0.0 <= score_val <= 10.0:
                                    scores[pend_matched] = max(scores.get(pend_matched, 0.0), score_val)
                                    matched_lines.append({
                                        "subject": pend_matched,
                                        "score": score_val,
                                        "ocr_text": pending_raw + " | " + full_line,
                                        "confidence": 1.0
                                    })
                                    pending_subj = ""
                                    pending_score = ""
                                    pending_raw = ""
                                    continue
                            except (ValueError, TypeError):
                                pass
                continue
                        
            # Nếu chưa xác định được cột Tên môn từ header, thử tìm dựa trên dữ liệu thực tế
            if subject_col_idx == -1:
                for i, cell_val in enumerate(lower_row):
                    if match_subject(cell_val):
                        subject_col_idx = i
                        break
                        
            # Xử lý cho bảng đã xác định được cột Tên môn học
            if subject_col_idx != -1 and subject_col_idx < len(row):
                cell_subj = str(row[subject_col_idx]).strip()
                cell_code = str(row[code_col_idx]).strip() if code_col_idx != -1 and code_col_idx < len(row) else ""
                cell_score = str(row[score_col_idx]).strip() if score_col_idx != -1 and score_col_idx < len(row) else ""
                raw_text = " | ".join(row)
                
                # Nối tiếp với dòng trước đó (xử lý môn bị cắt xuống dòng)
                combined_subj = (pending_subj + " " + cell_subj).strip()
                combined_score = cell_score if cell_score else pending_score
                combined_raw = (pending_raw + " | " + raw_text) if pending_raw else raw_text
                
                # 1. Thử match tên môn kết hợp (nếu có dòng trước chưa khớp)
                matched_subj = None
                if pending_subj:
                    matched_subj = match_subject(combined_subj)
                    
                # 2. Thử match nguyên ô Tên môn hiện tại
                if not matched_subj:
                    matched_subj = match_subject(cell_subj)
                    if matched_subj:
                        combined_subj = cell_subj
                        combined_score = cell_score
                        combined_raw = raw_text
                        
                # 3. Thử match theo ô Mã học phần
                if not matched_subj and cell_code:
                    matched_subj = match_subject(cell_code)
                    if matched_subj:
                        combined_subj = cell_subj
                        combined_score = cell_score
                        combined_raw = raw_text

                # 4. Thử quét toàn bộ các ô trong dòng
                if not matched_subj:
                    for cell in row:
                        m = match_subject(str(cell).strip())
                        if m:
                            matched_subj = m
                            combined_subj = cell_subj or str(cell).strip()
                            combined_score = cell_score
                            combined_raw = raw_text
                            break
                
                if matched_subj:
                    found_score = False
                    match = SCORE_PATTERN.search(combined_score)
                    # Nếu ô điểm theo cột không có số hợp lệ, quét các ô phía sau của dòng
                    if not match:
                        for cell in reversed(row):
                            m_candidate = SCORE_PATTERN.search(str(cell).strip())
                            if m_candidate:
                                try:
                                    val = float(m_candidate.group(1).replace(',', '.'))
                                    if 0.0 <= val <= 10.0:
                                        match = m_candidate
                                        break
                                except (ValueError, TypeError):
                                    pass

                    if match:
                        score_str = match.group(1).replace(',', '.')
                        try:
                            score = float(score_str)
                            if 0.0 <= score <= 10.0:
                                scores[matched_subj] = max(scores.get(matched_subj, 0.0), score)
                                matched_lines.append({
                                    "subject": matched_subj,
                                    "score": score,
                                    "ocr_text": combined_raw,
                                    "confidence": 1.0
                                })
                                found_score = True
                                # Reset sau khi thành công
                                pending_subj = ""
                                pending_score = ""
                                pending_raw = ""
                        except (ValueError, TypeError):
                            pass
                            
                    if not found_score:
                        # Match môn nhưng chưa có điểm -> Đợi dòng tiếp theo
                        pending_subj = combined_subj
                        pending_score = combined_score
                        pending_raw = combined_raw
                else:
                    # Không match môn -> Lưu lại chờ nối dòng
                    if cell_subj:
                        if len(pending_subj) < 100:
                            pending_subj = combined_subj
                            pending_score = combined_score
                            pending_raw = combined_raw
                        else:
                            unmatched_lines.append(pending_raw)
                            pending_subj = cell_subj
                            pending_score = cell_score
                            pending_raw = raw_text
            else:
                unmatched_lines.append(" | ".join(row))
                
        if not scores:
            warnings.append("⚠️ Không trích xuất được môn học nào từ PDF.")
            
    except Exception as e:
        warnings.append(f"Lỗi đọc PDF: {e}")
        
    return {
        "scores": scores,
        "raw_text": raw_texts,
        "matched_lines": matched_lines,
        "unmatched_lines": unmatched_lines,
        "warnings": warnings
    }

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_extraction()
