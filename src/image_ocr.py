"""
image_ocr.py — Trích xuất điểm sinh viên từ ảnh bảng điểm
===========================================================
Module sử dụng EasyOCR để đọc ảnh bảng điểm sinh viên FIT-HAU,
trích xuất tên môn học và điểm số, rồi map sang danh sách 24 môn
chuẩn của hệ thống.

Hỗ trợ:
    - Ảnh chụp bảng điểm giấy (điện thoại)
    - Ảnh chụp màn hình portal/website trường
    - Screenshot file PDF bảng điểm
"""

import re
import unicodedata
import numpy as np
from PIL import Image
import easyocr
import os

# ============================================================
# CONSTANTS — 24 MÔN HỌC CHUẨN (khớp với data_pipeline.py)
# ============================================================
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

# ============================================================
# SUBJECT KEYWORD MAPPING — Fuzzy matching từ OCR text
# ============================================================
# Mapping từ keyword (lowercase, có thể bị OCR sai 1-2 ký tự)
# sang tên chuẩn. Dùng longest-match-first strategy.
SUBJECT_KEYWORDS: dict[str, str] = {
    # --- AN NINH MẠNG ---
    "an ninh mạng": "AN NINH MẠNG",
    "an ninh mang": "AN NINH MẠNG",
    "ninh mạng": "AN NINH MẠNG",
    "ninh mang": "AN NINH MẠNG",
    # --- AN TOÀN VÀ BẢO MẬT HTTT ---
    "an toàn và bảo mật hệ thống thông tin": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn và bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toan va bao mat": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    "bảo mật httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "bao mat httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn": "AN TOÀN VÀ BẢO MẬT HTTT",
    # --- C# ---
    "lập trình c#": "C#",
    "lap trinh c#": "C#",
    "ngôn ngữ c# và công nghệ .net": "C#",
    "ngôn ngữ c# và công nghệ net": "C#",
    # --- CÔNG NGHỆ PHẦN MỀM ---
    "công nghệ phần mềm": "CÔNG NGHỆ PHẦN MỀM",
    "cong nghệ phan mem": "CÔNG NGHỆ PHẦN MỀM",
    "cong nghe phan mem": "CÔNG NGHỆ PHẦN MỀM",
    "cnpm": "CÔNG NGHỆ PHẦN MỀM",
    # --- CƠ SỞ DỮ LIỆU ---
    "cơ sở dữ liệu": "CƠ SỞ DỮ LIỆU",
    "co so du lieu": "CƠ SỞ DỮ LIỆU",
    "csdl": "CƠ SỞ DỮ LIỆU",
    # --- CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT ---
    "cấu trúc dữ liệu và giải thuật": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "cau truc du lieu va giai thuat": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "cấu trúc dữ liệu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "cau truc du lieu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "ctdl": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "giải thuật": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    # --- GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH ---
    "gis và quản lý đô thị": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "gis va quan ly do thi": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "gis": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    "đô thị thông minh": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    # --- HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU ---
    "hệ quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "he quan tri co so du lieu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU", # Chống nhầm thành CSDL
    "tri co so du lieu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "quản trị cơ sở": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "quan tri co so": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị csdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "he quan tri": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hqtcsdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    # --- HỆ ĐIỀU HÀNH LINUX ---
    "hệ điều hành linux": "HỆ ĐIỀU HÀNH LINUX",
    "he dieu hanh linux": "HỆ ĐIỀU HÀNH LINUX",
    "linux": "HỆ ĐIỀU HÀNH LINUX",
    # --- HỆ ĐIỀU HÀNH ---
    "hệ điều hành": "HỆ ĐIỀU HÀNH",
    "he dieu hanh": "HỆ ĐIỀU HÀNH",
    "hê điêu bành": "HỆ ĐIỀU HÀNH",
    "hê điêu banh": "HỆ ĐIỀU HÀNH",
    # --- JAVA ---
    "công nghệ java": "JAVA",
    "lập trình java": "JAVA",
    "lap trinh java": "JAVA",
    "java": "JAVA",
    # --- KIẾN TRÚC MÁY TÍNH ---
    "kiến trúc máy tính": "KIẾN TRÚC MÁY TÍNH",
    "kien truc may tinh": "KIẾN TRÚC MÁY TÍNH",
    "ktmt": "KIẾN TRÚC MÁY TÍNH",
    # --- KỸ THUẬT LẬP TRÌNH ---
    "kỹ thuật lập trình": "KỸ THUẬT LẬP TRÌNH",
    "ky thuat lap trinh": "KỸ THUẬT LẬP TRÌNH",
    "kỷ thuật lập trình": "KỸ THUẬT LẬP TRÌNH",
    "ky thuật lập trình": "KỸ THUẬT LẬP TRÌNH",
    "kỹ thuat lap trinh": "KỸ THUẬT LẬP TRÌNH",
    "kỹ thuật lạp trình": "KỸ THUẬT LẬP TRÌNH",
    "ktlt": "KỸ THUẬT LẬP TRÌNH",
    # --- KỸ THUẬT ĐỒ HOẠ MÁY TÍNH ---
    "kỹ thuật đồ họa máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ hoạ máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "ky thuat do hoa may tinh": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ hoạ": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "kỹ thuật đồ họa": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "đồ hoạ máy tính": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "do hoa may tinh": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "đồ họa": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    # --- LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG ---
    "lập trình hướng đối tượng": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "lap trinh huong doi tuong": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "hướng đối tượng": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "oop": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "lthdtg": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    # --- LẬP TRÌNH WEB ---
    "lập trình web": "LẬP TRÌNH WEB",
    "công nghệ web": "LẬP TRÌNH WEB",
    "cong nghe web": "LẬP TRÌNH WEB",
    # --- MẠNG MÁY TÍNH ---
    "mạng máy tính": "MẠNG MÁY TÍNH",
    "mang may tinh": "MẠNG MÁY TÍNH",
    # --- NHẬP MÔN CNTT VÀ TRUYỀN THÔNG ---
    "nhập môn cntt và truyền thông": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhap mon cntt va truyen thong": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhập môn cntt": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "nhap mon cntt": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    # --- PHÂN TÍCH VÀ THIẾT KẾ HTTT ---
    "phân tích và thiết kế hệ thống thông tin": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích và thiết kế httt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phan tich va thiet ke httt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích thiết kế": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích và thiết kế": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "pttkhttt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    # --- QUẢN TRỊ MẠNG MÁY TÍNH ---
    "quản trị mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quan tri mang may tinh": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "trị mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH", # Dài hơn chữ "mạng máy tính" (13) để chống đè
    "tri mang may tinh": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản tri mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mang máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mạng may tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mạng máy tinh": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản trị mạng": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quan tri mang": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "quản tri mạng": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "trị mạng": "QUẢN TRỊ MẠNG MÁY TÍNH", # Fallback cực mạnh
    "tri mang": "QUẢN TRỊ MẠNG MÁY TÍNH",
    # --- TOÁN RỜI RẠC ---
    "toán rời rạc": "TOÁN RỜI RẠC",
    "toan roi rac": "TOÁN RỜI RẠC",
    "rời rạc": "TOÁN RỜI RẠC",
    # --- TRÍ TUỆ NHÂN TẠO ---
    "trí tuệ nhân tạo": "TRÍ TUỆ NHÂN TẠO",
    "tri tue nhan tao": "TRÍ TUỆ NHÂN TẠO",
    "ttnt": "TRÍ TUỆ NHÂN TẠO",
    "nhân tạo": "TRÍ TUỆ NHÂN TẠO",
    # --- XỬ LÝ TÍN HIỆU SỐ ---
    "xử lý tín hiệu số": "XỬ LÝ TÍN HIỆU SỐ",
    "xu ly tin hieu so": "XỬ LÝ TÍN HIỆU SỐ",
    "tín hiệu số": "XỬ LÝ TÍN HIỆU SỐ",
    "xlths": "XỬ LÝ TÍN HIỆU SỐ",
    # --- XỬ LÝ ẢNH ---
    "xử lý ảnh": "XỬ LÝ ẢNH",
    "xu ly anh": "XỬ LÝ ẢNH",
    "kỹ thuật xử lý ảnh": "XỬ LÝ ẢNH",
    "ky thuat xu ly anh": "XỬ LÝ ẢNH",
    # --- SUBJECT CODES (Mã học phần) ---
    "th5201": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "th4303": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "th4304": "KỸ THUẬT LẬP TRÌNH",
    "th5203": "HỆ ĐIỀU HÀNH",
    "th4319": "KIẾN TRÚC MÁY TÍNH",
    "th4305": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "th5217": "CƠ SỞ DỮ LIỆU",
    "th4306": "CÔNG NGHỆ PHẦN MỀM",
    "th4320": "TRÍ TUỆ NHÂN TẠO",
    "th5206": "MẠNG MÁY TÍNH",
    "th4316": "JAVA",
    "th5208": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "th5221": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "th5210": "AN TOÀN VÀ BẢO MẬT HTTT",
    "th4315": "C#",
    "th5231": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "th5211": "HỆ ĐIỀU HÀNH LINUX",
    "th4309": "LẬP TRÌNH WEB",
    "th4318": "PHÁT TRIỂN PHẦN MỀM",
    "th5213": "LẬP TRÌNH MẠNG",
    "th5216": "ĐỒ HỌA VÀ HIỆN THỰC ẢO",
    "th5218": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "th5219": "AN NINH MẠNG",
    "th5205": "XỬ LÝ TÍN HIỆU SỐ",
    "th5302": "ĐỒ ÁN TỐT NGHIỆP",
    # --- PARTIAL KEYWORDS FALLBACK ---
    "cấu trúc dữ liệu": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "liệu và giải thuật": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "lập trình hướng đối": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "hướng đối": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "cơ sở dữ": "CƠ SỞ DỮ LIỆU",
    "trí tuệ nhân": "TRÍ TUỆ NHÂN TẠO",
    "mạng máy": "MẠNG MÁY TÍNH",
    "phân tích thiết kế": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "thiết kế hệ": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "bảo mật hệ thống": "AN TOÀN VÀ BẢO MẬT HTTT",
    "c# và công": "C# VÀ CÔNG NGHỆ .NET",
    "kỹ thuật đồ": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "hệ điều": "HỆ ĐIỀU HÀNH",
    "xử lý tín": "XỬ LÝ TÍN HIỆU SỐ",
    "kỹ năng": "KỸ NĂNG MỀM", # assuming TH5224 Kỹ năng QT
}

# Regex pattern: số thực (7.5, 8,3, 10.0) hoặc số nguyên (7, 8, 9, 10)
# \b(\d{1,2}(?:[.,]\d)?)\b
SCORE_PATTERN = re.compile(r'\b(\d{1,2}(?:[.,]\d)?)\b')


# ============================================================
# EasyOCR READER (singleton)
# ============================================================
_reader = None

def get_reader():
    """Lazy-load EasyOCR reader (tải model lần đầu, cache cho các lần sau)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['vi', 'en'], gpu=False)
    return _reader


# ============================================================
# CORE FUNCTIONS
# ============================================================
def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt."""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def normalize_text(text: str) -> str:
    """Chuẩn hoá OCR text: NFC, lowercase, bỏ ký tự thừa, gộp khoảng trắng."""
    text = unicodedata.normalize('NFC', text)
    text = text.lower().strip()
    # Bỏ các ký tự đặc biệt thường gặp do OCR sai
    text = re.sub(r'[_\-–—|/\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def match_subject(text: str) -> str | None:
    """Match OCR text sang tên môn chuẩn bằng keyword matching.
    
    Strategy: longest-match-first để tránh match sai.
    Ví dụ: "hệ điều hành linux" phải match trước "hệ điều hành".
    Hỗ trợ đối chiếu cả dạng có dấu và không dấu (nếu có dấu trượt).
    """
    normalized = normalize_text(text)
    normalized_no_accents = remove_accents(normalized)
    
    # Pass 0: Match Mã học phần (ưu tiên cao nhất, vì nó là định danh duy nhất)
    # Ví dụ: "th4305", "tc2611", v.v. Các mã có dạng 2 chữ cái + 4 số
    subject_codes = [kw for kw in SUBJECT_KEYWORDS.keys() if re.match(r'^[a-z]{2}\d{4}$', kw)]
    for code in subject_codes:
        if re.search(rf'\b{code}\b', normalized):
            return SUBJECT_KEYWORDS[code]
            
    # Sort keywords by length (descending) for longest-match-first
    sorted_keywords = sorted([k for k in SUBJECT_KEYWORDS.keys() if k not in subject_codes], key=len, reverse=True)
    
    # Pass 1: Match có dấu (chính xác hơn)
    for keyword in sorted_keywords:
        if keyword in normalized:
            return SUBJECT_KEYWORDS[keyword]
            
    # Pass 2: Match không dấu (để chống lỗi OCR sai dấu)
    for keyword in sorted_keywords:
        keyword_no_accents = remove_accents(keyword)
        if keyword_no_accents in normalized_no_accents:
            return SUBJECT_KEYWORDS[keyword]
    
    return None


def extract_score_from_text(text: str) -> float | None:
    """Trích xuất điểm TBCHP (Trung bình chung học phần) từ chuỗi text.
    
    Phân tích cấu trúc bảng điểm HAU:
    <Mã HP> <Tên môn> <Số TC> <Lần học> QT : <QT> <Thi> <TBCHP> <Hệ 4> <Chữ>
    
    Quy tắc an toàn:
    - Nếu dòng không có điểm (môn chưa học, chỉ có Số TC và Lần học) -> trả về None
    - Tuyệt đối không lấy nhầm cột Số TC, Lần học, hay Điểm thi làm TBCHP.
    """
    score_pattern = re.compile(r'(?<![\d.,])(\d{1,2}(?:[.,]\d)?)(?![\d.,])')
    
    # 1. Tìm theo neo "QT" (Điểm quá trình)
    qt_pattern = re.compile(r'(?:Q\s*T|0\s*T|O\s*T|Q\s*7|Q\s*I)\s*[:;.\-]?\s*', re.IGNORECASE)
    qt_matches = list(qt_pattern.finditer(text))
    
    if qt_matches:
        last_qt = qt_matches[-1]
        after_qt = text[last_qt.end():]
        after_scores = []
        for m in score_pattern.findall(after_qt):
            try:
                sc = float(m.replace(',', '.'))
                if 0.0 <= sc <= 10.0:
                    after_scores.append(sc)
            except ValueError:
                continue
                
        # Sau QT thứ tự là: QT (nếu nằm sau chữ QT), Thi, TBCHP, Hệ 4
        # Thường có 3 số trở lên: [QT, Thi, TBCHP, ...]
        if len(after_scores) >= 3:
            return after_scores[2]  # Điểm TBCHP là số thứ 3
        elif len(after_scores) == 2:
            # Nếu chỉ có 2 số sau QT, thường là [Thi, TBCHP] hoặc [QT, TBCHP]
            return after_scores[1]
        elif len(after_scores) == 1:
            # Chỉ có 1 số sau QT -> Đây là điểm QT, CHƯA CÓ ĐIỂM THI VÀ TBCHP!
            return None
            
    # 2. Nếu KHÔNG có từ khoá QT (hoặc neo điểm)
    # Kiểm tra xem có từ khoá điểm khác không: "tbchp", "tbcmh", "tổng kết", "điểm"
    has_score_keyword = bool(re.search(r'\b(?:tbchp|tbcmh|tổng kết|điểm|thi)\b', text, re.IGNORECASE))
    
    matches = score_pattern.findall(text)
    valid_scores = []
    has_decimal = False
    for match in matches:
        try:
            if '.' in match or ',' in match:
                has_decimal = True
            score = float(match.replace(',', '.'))
            if 0.0 <= score <= 10.0:
                valid_scores.append(score)
        except ValueError:
            continue
            
    if not valid_scores:
        return None
        
    # CHỐNG FALSE POSITIVE CHO MÔN CHƯA HỌC:
    # Nếu không có từ khoá điểm và không có số thập phân:
    # Ví dụ dòng chỉ có: "GIS và quản lý đô thị 3 1" -> valid_scores = [3.0, 1.0]
    # Đây là [Số TC, Lần học], HOÀN TOÀN KHÔNG PHẢI ĐIỂM SỐ!
    if not has_score_keyword:
        if len(valid_scores) <= 2 and not has_decimal and all(s <= 4.0 for s in valid_scores):
            return None
        if len(valid_scores) < 3 and not has_decimal:
            return None
            
    if len(valid_scores) >= 3:
        # Nếu có từ 3 số trở lên, số áp chót thường là TBCHP (hoặc số thứ 3 từ dưới lên nếu có điểm chữ OCR nhầm)
        last = valid_scores[-1]
        second_last = valid_scores[-2]
        third_last = valid_scores[-3]
        if last <= 4.0:
            return second_last  # second_last là TBCHP, last là Hệ 4
        elif second_last <= 4.0 and last > 4.0:
            return third_last
        return second_last
        
    return None


def extract_scores_from_image(image_data) -> dict:
    """Trích xuất điểm số 24 môn học từ ảnh bảng điểm (hỗ trợ ảnh portal, scan PDF, giấy).
    
    Ưu tiên tuyệt đối:
    1. Lấy chính xác cột điểm TBCHP (sử dụng toạ độ không gian cột TBCHP trong bảng).
    2. Chống nhận diện nhầm các môn chưa có điểm (trả về None / không đưa vào danh sách có điểm).
    """
    reader = get_reader()
    
    # Chuẩn hoá sang PIL Image để tiền xử lý
    if isinstance(image_data, str):
        pil_img = Image.open(image_data)
    elif isinstance(image_data, np.ndarray):
        pil_img = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        pil_img = image_data
    else:
        pil_img = Image.open(image_data)
        
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
        
    w_orig, h_orig = pil_img.size
    
    # ── Bước 1: Upscale bằng Lanczos để EasyOCR nhận diện rõ từng ô số nhỏ và số đơn lẻ ──
    scale = 2.0 if w_orig < 1400 else 1.5
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img_scaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    from PIL import ImageEnhance
    img_scaled = ImageEnhance.Contrast(img_scaled).enhance(1.25)
    
    # ── Bước 2: OCR với tham số tối ưu cho bảng điểm ──
    results = reader.readtext(
        np.array(img_scaled),
        detail=1,
        paragraph=False,
        text_threshold=0.3,
        low_text=0.2,
        link_threshold=0.25,
        mag_ratio=1.5
    )
    
    scores: dict[str, float] = {}
    raw_texts: list[str] = []
    matched_lines: list[dict] = []
    unmatched_lines: list[str] = []
    warnings: list[str] = []
    
    # Chuẩn hoá danh sách items có toạ độ tương đối x_rel, y_rel
    items = []
    for bbox, text, conf in results:
        raw_texts.append(text)
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({
            "text": text.strip(),
            "conf": conf,
            "x_min": min(xs),
            "x_max": max(xs),
            "x_mid": (min(xs) + max(xs)) / 2,
            "y_min": min(ys),
            "y_max": max(ys),
            "y_mid": (min(ys) + max(ys)) / 2,
            "x_rel": ((min(xs) + max(xs)) / 2) / new_w,
            "y_rel": ((min(ys) + max(ys)) / 2) / new_h,
        })
        
    # ── Bước 3: Gom nhóm theo dòng theo toạ độ Y-mid ──
    items.sort(key=lambda it: it["y_mid"])
    y_thresh = 18 * scale
    rows = []
    for it in items:
        placed = False
        for r in rows:
            r_y_mid = sum(x["y_mid"] for x in r) / len(r)
            if abs(it["y_mid"] - r_y_mid) <= y_thresh:
                r.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])
            
    score_regex = re.compile(r'(?<![\d.,])(\d{1,2}(?:[.,]\d)?)(?![\d.,])')
    
    # ── Bước 4: Duyệt từng dòng, ghép dòng nối tiếp và trích xuất điểm TBCHP ──
    pending_row_items = []
    
    for r in rows:
        r.sort(key=lambda it: it["x_min"])
        full_line_text = " ".join([it['text'] for it in r])
        
        # Thử xem dòng hiện tại có chứa môn học không
        direct_subject = match_subject(full_line_text)
        
        if direct_subject:
            combined_items = r
            subject = direct_subject
            pending_row_items = []
        else:
            # Thử ghép với dòng chờ phía trước (xử lý tên môn bị ngắt dòng)
            combined_items = pending_row_items + r
            combined_text = " ".join([it['text'] for it in combined_items])
            subject = match_subject(combined_text)
            
        if subject:
            # ĐÃ MATCH ĐƯỢC MÔN HỌC!
            extracted_score = None
            
            # --- Chiến lược A: Dựa vào cột toạ độ không gian chuẩn của bảng portal HAU ---
            # Trong bảng điểm portal HAU:
            # x_rel < 0.45: STT, Mã HP, Tên môn, Số TC, Lần học
            # x_rel in [0.46, 0.54]: QT (Quá trình)
            # x_rel in [0.54, 0.59]: Thi
            # x_rel in [0.58, 0.67]: CỘT ĐIỂM TBCHP CHUẨN!
            # x_rel in [0.67, 0.74]: Điểm hệ 4
            # x_rel >= 0.74: Điểm chữ (A, B, C...)
            tbchp_candidates = []
            for it in combined_items:
                if 0.58 <= it["x_rel"] <= 0.67:
                    for m in score_regex.findall(it["text"]):
                        try:
                            val = float(m.replace(',', '.'))
                            if 0.0 <= val <= 10.0:
                                tbchp_candidates.append(val)
                        except ValueError:
                            pass
                            
            if tbchp_candidates:
                extracted_score = tbchp_candidates[0]
            else:
                # --- Chiến lược B: Fallback dựa vào từ khoá QT và vị trí số ---
                has_qt = any(re.search(r'(?:Q\s*T|0\s*T|O\s*T|Q\s*7|Q\s*I)', it["text"], re.IGNORECASE) for it in combined_items)
                score_items = [it for it in combined_items if it["x_rel"] >= 0.48]
                
                # BẢO VỆ CHỐNG TÍCH NHẦM: Nếu không có QT và không có ô nào ở vùng điểm, môn này CHƯA CÓ ĐIỂM!
                if not has_qt and not score_items:
                    extracted_score = None
                else:
                    row_full_text = " ".join([it['text'] for it in combined_items])
                    extracted_score = extract_score_from_text(row_full_text)
                    
            if extracted_score is not None:
                if subject not in scores or extracted_score > scores[subject]:
                    scores[subject] = extracted_score
                    
                matched_lines.append({
                    "subject": subject,
                    "score": extracted_score,
                    "ocr_text": " ".join([it['text'] for it in combined_items]),
                    "confidence": min([it['conf'] for it in combined_items]) if combined_items else 1.0,
                })
                pending_row_items = []
            else:
                # Môn này không có điểm (chưa học) -> Reset pending để không dính sang dòng sau
                pending_row_items = []
        else:
            # Dòng không match được môn nào -> lưu lại chờ ghép nếu dòng ngắn (< 50 ký tự)
            if len(full_line_text) < 50:
                pending_row_items = r
            else:
                unmatched_lines.append(full_line_text)
                pending_row_items = []
                
    # ============================================================
    # Warnings
    # ============================================================
    if not scores:
        warnings.append(
            "⚠️ Không trích xuất được điểm nào từ ảnh. "
            "Vui lòng kiểm tra: (1) Ảnh có rõ nét không? "
            "(2) Ảnh có chứa bảng điểm với tên môn và điểm số không?"
        )
    elif len(scores) < 5:
        warnings.append(
            f"⚠️ Chỉ trích xuất được {len(scores)}/24 môn. "
            "Kết quả có thể không chính xác. "
            "Bạn nên kiểm tra lại các điểm đã trích xuất."
        )
    
    return {
        "scores": scores,
        "raw_text": raw_texts,
        "matched_lines": matched_lines,
        "unmatched_lines": unmatched_lines,
        "warnings": warnings,
    }


def build_full_score_dict(extracted_scores: dict[str, float]) -> dict[str, float]:
    """Xây dựng dict 24 môn đầy đủ từ điểm đã trích xuất.
    
    Các môn không có điểm sẽ được gán -1.0 (chưa học).
    
    Args:
        extracted_scores: dict từ extract_scores_from_image()['scores']
        
    Returns:
        dict[str, float] — 24 môn với giá trị điểm hoặc -1.0
    """
    full_scores = {}
    for subject in STANDARD_SUBJECTS:
        if subject in extracted_scores:
            full_scores[subject] = extracted_scores[subject]
        else:
            full_scores[subject] = -1.0
    return full_scores



