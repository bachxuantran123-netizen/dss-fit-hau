"""
subject_matcher.py — Subject matching & text normalization for student transcript PDFs
=====================================================================================
Provides helper functions for matching Vietnamese subject names in text
extracted from individual student transcript PDFs.

Functions:
    normalize_text  — Strip Vietnamese diacritics, lowercase, normalize whitespace.
    match_subject   — Longest-match-first keyword lookup against 24 IT subjects.
    SCORE_PATTERN   — Compiled regex to capture decimal scores (e.g., "7.5", "8,0").
"""

import re
import unicodedata


# ============================================================
# CONSTANTS
# ============================================================

# Regex pattern: captures scores in format "7.5", "8,0", "10.0", etc.
# Group(1) = the numeric score string (may use comma as decimal separator)
SCORE_PATTERN: re.Pattern = re.compile(r'(\d{1,2}[.,]\d+)')

# Mapping: normalized (ASCII, lowercase) keyword → standardized Vietnamese subject name
# Sorted by key length descending at lookup time (longest match first)
_SUBJECT_KEYWORDS: dict[str, str] = {
    # --- Longer (more specific) keywords first to avoid partial matches ---
    "he quan tri co so du lieu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "cau truc du lieu va giai thuat": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "gis va quan ly do thi thong minh": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "nhap mon cong nghe thong tin va truyen thong": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhap mon cong nghe thong tin": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "an toan va bao mat httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "phan tich va thiet ke httt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "lap trinh huong doi tuong": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "quan tri mang may tinh": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "ky thuat do hoa may tinh": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "xu ly tin hieu so": "XỬ LÝ TÍN HIỆU SỐ",
    "he dieu hanh linux": "HỆ ĐIỀU HÀNH LINUX",
    "ky thuat xu ly anh": "XỬ LÝ ẢNH",
    "an toan va bao mat": "AN TOÀN VÀ BẢO MẬT HTTT",
    "phan tich va thiet ke": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "cong nghe phan mem": "CÔNG NGHỆ PHẦN MỀM",
    "kien truc may tinh": "KIẾN TRÚC MÁY TÍNH",
    "ky thuat lap trinh": "KỸ THUẬT LẬP TRÌNH",
    "cau truc du lieu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "mang may tinh": "MẠNG MÁY TÍNH",
    "an ninh mang": "AN NINH MẠNG",
    "do hoa may tinh": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "ky thuat do hoa": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "tri tue nhan tao": "TRÍ TUỆ NHÂN TẠO",
    "xu ly tin hieu": "XỬ LÝ TÍN HIỆU SỐ",
    "co so du lieu": "CƠ SỞ DỮ LIỆU",
    "gis va quan ly": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "he dieu hanh": "HỆ ĐIỀU HÀNH",
    "he quan tri": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "quan tri mang": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "lap trinh java": "JAVA",
    "lap trinh web": "LẬP TRÌNH WEB",
    "lap trinh c#": "C#",
    "nhap mon cntt": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "toan roi rac": "TOÁN RỜI RẠC",
    "xu ly anh": "XỬ LÝ ẢNH",
    "java": "JAVA",
    "c#": "C#",
}


# ============================================================
# FUNCTIONS
# ============================================================
def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for fuzzy matching.

    Steps:
        1. Replace đ/Đ with d (manual — NFKD does not decompose đ).
        2. NFKD decompose and strip combining marks (removes diacritics).
        3. Lowercase.
        4. Collapse whitespace.
    """
    # đ does not decompose via NFKD, handle manually
    text = text.replace('đ', 'd').replace('Đ', 'd')
    # Decompose and strip combining marks (tonal + diacritical)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase and normalize whitespace
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r'\s+', ' ', ascii_text).strip()
    return ascii_text


def match_subject(text: str) -> str | None:
    """Match normalized text to a standardized subject name.

    Uses longest-match-first strategy to avoid partial matches
    (e.g., 'he dieu hanh linux' must match before 'he dieu hanh').

    Args:
        text: Normalized (ASCII, lowercase) text from normalize_text().

    Returns:
        Standardized Vietnamese subject name, or None if no match.
    """
    # Sort by key length descending for longest match first
    for keyword in sorted(_SUBJECT_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in text:
            return _SUBJECT_KEYWORDS[keyword]
    return None
