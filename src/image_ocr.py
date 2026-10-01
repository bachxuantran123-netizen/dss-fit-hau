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
    """Trích xuất điểm số (0.0 - 10.0) từ chuỗi OCR text.
    
    Hỗ trợ cả dấu chấm (7.5), dấu phẩy (7,5) và số nguyên (9, 10).
    Sẽ lấy con số hợp lệ (0-10) xuất hiện *cuối cùng* trong chuỗi, 
    vì điểm tổng kết thường nằm ở cuối dòng.
    """
    matches = SCORE_PATTERN.findall(text)
    
    valid_scores = []
    for match in matches:
        try:
            score = float(match.replace(',', '.'))
            if 0.0 <= score <= 10.0:
                valid_scores.append(score)
        except ValueError:
            continue
            
    # Lấy điểm hợp lệ xuất hiện cuối cùng trong dòng
    if valid_scores:
        return valid_scores[-1]
    
    return None


def extract_scores_from_image(image_data) -> dict:
    """Trích xuất điểm số 24 môn học từ ảnh bảng điểm.
    
    Args:
        image_data: PIL Image, numpy array, hoặc đường dẫn file ảnh.
        
    Returns:
        dict với keys:
            - 'scores': dict[str, float] — Điểm trích xuất được (tên môn chuẩn -> điểm)
            - 'raw_text': list[str] — Toàn bộ text OCR đọc được (để debug)
            - 'matched_lines': list[dict] — Chi tiết các dòng đã match thành công
            - 'unmatched_lines': list[str] — Các dòng không match được môn nào
            - 'warnings': list[str] — Cảnh báo (nếu có)
    """
    reader = get_reader()
    
    # Chuyển PIL Image sang numpy array nếu cần
    if isinstance(image_data, Image.Image):
        image_data = np.array(image_data)
    
    # OCR
    results = reader.readtext(image_data, detail=1, paragraph=False)
    
    # Kết quả
    scores: dict[str, float] = {}
    raw_texts: list[str] = []
    matched_lines: list[dict] = []
    unmatched_lines: list[str] = []
    warnings: list[str] = []
    
    # ============================================================
    # STRATEGY 1: Group theo dòng (Y-coordinate gần nhau)
    # ============================================================
    # Sắp xếp theo Y-coordinate (top-to-bottom)
    sorted_results = sorted(results, key=lambda r: r[0][0][1])  # sort by top-left Y
    
    # Group các text box cùng dòng (Y-diff < 20px)
    lines: list[list] = []
    current_line: list = []
    current_y: float = -100
    
    for bbox, text, conf in sorted_results:
        raw_texts.append(text)
        top_y = bbox[0][1]  # top-left Y
        
        if abs(top_y - current_y) > 20:
            # Dòng mới
            if current_line:
                lines.append(current_line)
            current_line = [(bbox, text, conf)]
            current_y = top_y
        else:
            current_line.append((bbox, text, conf))
    
    if current_line:
        lines.append(current_line)
    
    # ============================================================
    # STRATEGY 2: Duyệt từng dòng, tìm tên môn + điểm
    # ============================================================
    pending_text = ""
    pending_conf = 1.0
    
    for line_items in lines:
        # Sắp xếp items trong dòng theo X (left-to-right)
        line_items.sort(key=lambda item: item[0][0][0])
        
        # Gộp text của dòng
        full_line_text = " ".join([item[1] for item in line_items])
        line_conf = min([item[2] for item in line_items]) if line_items else 1.0
        
        # Nối với text chờ trước đó (nếu có)
        combined_text = (pending_text + " " + full_line_text).strip()
        combined_conf = min(pending_conf, line_conf)
        
        # Thử xem dòng mới có chứa một môn học HOÀN TOÀN KHÁC không
        # Nếu có, ta không nên nối vào pending_text cũ vì đã sang môn mới
        new_line_subject = match_subject(full_line_text)
        
        # Thử match trên chuỗi đã nối
        subject = match_subject(combined_text)
        
        # Nếu dòng mới là một môn khác, và chuỗi nối vẫn match môn cũ,
        # có nghĩa là ta đã ghép nhầm 2 môn với nhau.
        if new_line_subject and pending_text:
            pending_subj = match_subject(pending_text)
            if pending_subj and new_line_subject != pending_subj:
                # Đã sang môn mới! Reset chuỗi nối
                combined_text = full_line_text
                combined_conf = line_conf
                subject = new_line_subject
        
        if not subject:
            # Nếu chuỗi nối không match, thử match riêng lẻ (trường hợp chuỗi pending là rác)
            subject = new_line_subject
            if subject:
                combined_text = full_line_text
                combined_conf = line_conf
                
        if subject:
            # Tìm điểm
            score = extract_score_from_text(combined_text)
            
            if score is not None:
                if subject not in scores or score > scores[subject]:
                    scores[subject] = score
                
                matched_lines.append({
                    "subject": subject,
                    "score": score,
                    "ocr_text": combined_text,
                    "confidence": combined_conf,
                })
                # Đã extract xong thì reset pending
                pending_text = ""
                pending_conf = 1.0
            else:
                # Tìm thấy môn nhưng không có điểm -> lưu lại chờ dòng tiếp theo
                pending_text = combined_text
                pending_conf = combined_conf
        else:
            # Không match được môn nào
            score_in_line = extract_score_from_text(full_line_text)
            if score_in_line is not None and len(full_line_text) > 5:
                unmatched_lines.append(full_line_text)
                
            # Lưu dòng hiện tại để chờ ghép với dòng tiếp theo
            if len(pending_text) < 100:
                pending_text = combined_text
                pending_conf = combined_conf
            else:
                pending_text = full_line_text
                pending_conf = line_conf
    
    # Đã xoá Strategy 3 vì gây ra rất nhiều lỗi nhận diện nhầm điểm (false positives).
    
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



