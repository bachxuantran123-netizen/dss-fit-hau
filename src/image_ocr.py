"""
image_ocr.py (v2) — Trích xuất điểm sinh viên từ ảnh / PDF bảng điểm
====================================================================
Dành cho bảng điểm của Cổng thông tin sinh viên FIT-HAU (tinchi.hau.edu.vn),
đồng thời vẫn có chế độ dự phòng cho ảnh chụp bảng điểm giấy.

Thuật toán chính (khác bản cũ):
    1. OCR -> danh sách ô chữ kèm toạ độ.
    2. Tìm các ô "Ký hiệu" (mã học phần, vd TH4302) làm NEO của từng hàng.
    3. Gán mọi ô chữ còn lại vào hàng có neo gần nhất theo trục Y
       (nên tên môn xuống dòng nhiều tầng không làm vỡ hàng).
    4. Mã học phần -> môn chuẩn (tra bảng mã, chính xác tuyệt đối).
       Chỉ khi mã lạ mới dò theo tên môn.
    5. Điểm lấy là TBCHP (hệ 10): cặp số [TBCHP, điểm hệ 4] ở cuối hàng,
       tự kiểm tra chéo bằng thang quy đổi 10 -> 4.
    6. Bỏ qua toàn bộ bảng "Danh sách học phần chưa học hoặc chưa có điểm".
"""

from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
from PIL import Image

# ============================================================
# 24 MÔN CHUẨN (khớp với data_pipeline.py)
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
_STANDARD_SET = set(STANDARD_SUBJECTS)

# ============================================================
# MÃ HỌC PHẦN -> MÔN CHUẨN (chỉ gồm đúng 24 môn)
# ============================================================
# Bản cũ thiếu TH4302 (Toán rời rạc) và TH5209 (Xử lý ảnh) nên mất 2 môn,
# đồng thời map ra cả môn ngoài danh sách (TH5302, TH4318, ...).
SUBJECT_CODES: dict[str, str] = {
    "TH5219": "AN NINH MẠNG",
    "TH5210": "AN TOÀN VÀ BẢO MẬT HTTT",
    "TH4315": "C#",
    "TH4306": "CÔNG NGHỆ PHẦN MỀM",
    "TH5217": "CƠ SỞ DỮ LIỆU",
    "TH4303": "CẤU TRÚC DỮ LIỆU VÀ GIẢI THUẬT",
    "DT1926": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",   # DT1926.1
    "TH5221": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "TH5203": "HỆ ĐIỀU HÀNH",
    "TH5211": "HỆ ĐIỀU HÀNH LINUX",
    "TH4316": "JAVA",
    "TH4319": "KIẾN TRÚC MÁY TÍNH",
    "TH4304": "KỸ THUẬT LẬP TRÌNH",
    "TH5231": "KỸ THUẬT ĐỒ HOẠ MÁY TÍNH",
    "TH4305": "LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG",
    "TH4309": "LẬP TRÌNH WEB",                     # "Công nghệ Web"
    "TH5206": "MẠNG MÁY TÍNH",
    "TH5201": "NHẬP MÔN CNTT VÀ TRUYỀN THÔNG",
    "TH5208": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "TH5218": "QUẢN TRỊ MẠNG MÁY TÍNH",
    "TH4302": "TOÁN RỜI RẠC",
    "TH4320": "TRÍ TUỆ NHÂN TẠO",
    "TH5205": "XỬ LÝ TÍN HIỆU SỐ",
    "TH5209": "XỬ LÝ ẢNH",
}
assert set(SUBJECT_CODES.values()) == _STANDARD_SET, "SUBJECT_CODES phải phủ đúng 24 môn"

# ============================================================
# TÊN MÔN -> MÔN CHUẨN (dùng khi mã bị OCR sai / ảnh không có cột mã)
# ============================================================
_ALIASES_RAW: dict[str, str] = {
    # An ninh mạng
    "an ninh mạng": "AN NINH MẠNG",
    # ATBM
    "an toàn và bảo mật hệ thống thông tin": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn và bảo mật httt": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    "an toàn và bảo mật": "AN TOÀN VÀ BẢO MẬT HTTT",
    # C#
    "ngôn ngữ c# và công nghệ .net": "C#",
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
    "gis": "GIS VÀ QUẢN LÝ ĐÔ THỊ THÔNG MINH",
    # HQTCSDL
    "hệ quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hệ quản trị csdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "quản trị cơ sở dữ liệu": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
    "hqtcsdl": "HỆ QUẢN TRỊ CƠ SỞ DỮ LIỆU",
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
    # PTTKHT
    "phân tích và thiết kế hệ thống thông tin": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích và thiết kế httt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "phân tích thiết kế hệ thống": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    "pttkhttt": "PHÂN TÍCH VÀ THIẾT KẾ HTTT",
    # QTMMT
    "quản trị mạng máy tính": "QUẢN TRỊ MẠNG MÁY TÍNH",
    # Toán rời rạc
    "toán rời rạc": "TOÁN RỜI RẠC",
    # TTNT
    "trí tuệ nhân tạo": "TRÍ TUỆ NHÂN TẠO",
    "ttnt": "TRÍ TUỆ NHÂN TẠO",
    # XLTHS
    "xử lý tín hiệu số": "XỬ LÝ TÍN HIỆU SỐ",
    "xlths": "XỬ LÝ TÍN HIỆU SỐ",
    # Xử lý ảnh
    "xử lý ảnh": "XỬ LÝ ẢNH",
    "kỹ thuật xử lý ảnh": "XỬ LÝ ẢNH",
}


# ============================================================
# TIỆN ÍCH CHUỖI
# ============================================================
def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt (O(n), xử lý cả đ/Đ)."""
    s = unicodedata.normalize("NFD", input_str)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    """NFC, lowercase, bỏ ký tự phân cách do OCR, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[_\-–—|/\\]", " ", text)
    return re.sub(r"\s+", " ", text)


def _norm_key(text: str) -> str:
    """Khoá so khớp: không dấu, chữ thường, chỉ giữ [a-z0-9#] và khoảng trắng."""
    s = remove_accents(unicodedata.normalize("NFC", text).lower())
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ALIASES: dict[str, str] = {_norm_key(k): v for k, v in _ALIASES_RAW.items()}
_ALIASES_SORTED: list[str] = sorted(_ALIASES, key=len, reverse=True)  # dài trước
_FUZZY_KEYS: list[str] = [k for k in _ALIASES if len(k) >= 8]


def match_subject(text: str) -> str | None:
    """Map TÊN môn (OCR) sang môn chuẩn. Trả None nếu không thuộc 24 môn.

    1) khớp chính xác  2) khớp chuỗi con theo ranh giới từ, dài nhất trước
    3) khớp mờ (difflib) để chịu lỗi OCR 1-2 ký tự.
    """
    n = _norm_key(text)
    if not n:
        return None
    if n in _ALIASES:
        return _ALIASES[n]
    for alias in _ALIASES_SORTED:
        if re.search(rf"(?<![a-z0-9#]){re.escape(alias)}(?![a-z0-9#])", n):
            return _ALIASES[alias]
    if len(n) >= 8:
        close = difflib.get_close_matches(n, _FUZZY_KEYS, n=1, cutoff=0.86)
        if close:
            return _ALIASES[close[0]]
    return None


# ============================================================
# TÁCH ĐIỂM
# ============================================================
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_QT_RE = re.compile(r"[QO0]T\s*[:;.]?\s*\d+(?:[.,]\d+)?", re.I)   # "QT : 7.6" = điểm quá trình, bỏ

# Public regex — used by pdf_extractor.py to extract a single score from a text line.
# Captures the LAST number that looks like a score (0.0–10.0).
SCORE_PATTERN = re.compile(r"(\d{1,2}[.,]\d+|\d{1,2})\s*$")


def grade4_from_10(x: float) -> int:
    """Quy đổi thang 10 -> thang 4 theo quy chế HAU."""
    if x >= 8.5:
        return 4
    if x >= 7.0:
        return 3
    if x >= 5.5:
        return 2
    if x >= 4.0:
        return 1
    return 0


def _numbers_in(text: str) -> list[float]:
    text = _QT_RE.sub(" ", text)
    out: list[float] = []
    for tok in _NUM_RE.findall(text.replace(",", ".")):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _repair(v: float) -> float:
    """OCR hay làm rơi dấu chấm: 83 -> 8.3, 100 -> 10."""
    if float(v).is_integer():
        iv = int(v)
        if 40 <= iv <= 99:
            return iv / 10.0
        if iv == 100:
            return 10.0
    return v


def pick_tbchp(tokens: list[float]) -> float | None:
    """Tìm cặp [TBCHP, điểm hệ 4] RIGHTMOST trong dãy số của một hàng.

    Hàng chuẩn: ... | điểm thi | TBCHP | điểm hệ 4 | (điểm chữ)
    Cặp hợp lệ khi hệ 4 là số nguyên 0..4 và đúng bằng quy đổi của TBCHP.
    Hàng học lại ("3 | 7.2" ... "F | B") vẫn đúng vì lấy cặp ngoài cùng bên phải.
    """
    for repair in (False, True):
        seq = [_repair(t) for t in tokens] if repair else list(tokens)
        for j in range(len(seq) - 1, 0, -1):
            h4, t = seq[j], seq[j - 1]
            if float(h4).is_integer() and 0 <= h4 <= 4 and 0 <= t <= 10:
                if grade4_from_10(t) == int(h4):
                    return float(t)
    return None


def _numbers_with_pos(text: str) -> list[tuple[float, float]]:
    """Như _numbers_in nhưng kèm vị trí tương đối (0..1) của số trong chuỗi."""
    text = _QT_RE.sub(lambda m: " " * len(m.group()), text).replace(",", ".")
    n = max(len(text), 1)
    out = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append((float(m.group()), (m.start() + m.end()) / 2 / n))
        except ValueError:
            pass
    return out


def pick_row_score(toks: list[tuple], col_x: float | None = None,
                   tol: float | None = None, allow_column: bool = True) -> tuple[float | None, str, float | None]:
    """Chọn TBCHP của một hàng. toks = [(giá trị, cx ước lượng, cx của ô, số lượng số trong ô)].

    - Ưu tiên cặp [TBCHP, hệ 4] hợp lệ (kiểm tra chéo bằng thang 10->4).
    - Nếu biết vị trí cột TBCHP: số được chọn phải nằm đúng cột đó (chặn việc OCR bỏ sót
      ô TBCHP rồi vô tình ghép nhầm điểm thi + hệ 4). Ô EasyOCR gộp >= 3 số thì không
      định vị chính xác được nên không bị loại bởi kiểm tra này.
    - Không chắc chắn -> (None, "none", None), không đoán bừa.
    Trả về (điểm, phương pháp, cx của ô chứa điểm).
    """
    def col_ok(t) -> bool:
        if col_x is None:
            return True
        if t[3] >= 3:        # ô gộp nhiều số: không định vị được -> yêu cầu hàng đủ >= 5 số
            return len(toks) >= 5   # (tín chỉ, hệ số, điểm thi, TBCHP, hệ 4); thiếu số thì từ chối
        return min(abs(t[1] - col_x), abs(t[2] - col_x)) <= tol

    for repair in (False, True):
        seq = [((_repair(t[0]) if repair else t[0]),) + tuple(t[1:]) for t in toks]
        for j in range(len(seq) - 1, 0, -1):
            h4, t = seq[j][0], seq[j - 1]
            if (float(h4).is_integer() and 0 <= h4 <= 4 and 0 <= t[0] <= 10
                    and grade4_from_10(t[0]) == int(h4) and col_ok(t)):
                return float(t[0]), "pair", t[2]

    if col_x is not None and allow_column:      # hệ 4 bị mất nhưng vẫn đọc được đúng cột TBCHP
        # không "sửa dấu chấm" các số chia hết cho 15 (có thể là số tiết: 45, 60, 75, 90...)
        cand = [((_repair(t[0]) if (t[0] > 10 and t[0] % 15 != 0) else t[0]), t[2]) for t in toks
                if col_ok(t)]
        cand = [c for c in cand if 0 <= c[0] <= 10]
        if cand:
            return float(cand[-1][0]), "column", cand[-1][1]   # học lại: lấy giá trị sau cùng
    return None, "none", None


def extract_score_from_text(text: str) -> float | None:
    """Tách điểm TBCHP từ một chuỗi (giữ tên hàm cũ để tương thích)."""
    nums = _numbers_in(text)
    if not nums:
        return None
    s = pick_tbchp(nums)
    if s is not None:
        return s
    # dự phòng kiểu cũ: số cuối là hệ 4 -> lấy số liền trước
    if len(nums) >= 2 and float(nums[-1]).is_integer() and 0 <= nums[-1] <= 4 and 0 <= nums[-2] <= 10:
        return nums[-2]
    return nums[-1] if 0 <= nums[-1] <= 10 else None


# ============================================================
# OCR
# ============================================================
_reader = None


def get_reader():
    """Lazy-load EasyOCR (import trong hàm để module vẫn import được khi test)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["vi", "en"], gpu=False)
    return _reader


def _prepare_image(image_data) -> np.ndarray:
    """Nhận đường dẫn / PIL / ndarray -> ndarray RGB, phóng to nếu ảnh quá nhỏ."""
    if isinstance(image_data, (str, bytes)) or hasattr(image_data, "__fspath__"):
        image_data = Image.open(image_data)
    if isinstance(image_data, np.ndarray):
        image_data = Image.fromarray(image_data)
    img = image_data.convert("RGB")
    if img.width < 1400:                      # chữ nhỏ -> OCR kém
        scale = 1600 / img.width
        img = img.resize((1600, int(img.height * scale)), Image.LANCZOS)
    return np.array(img)


def _run_ocr(image_data) -> list:
    reader = get_reader()
    return reader.readtext(_prepare_image(image_data), detail=1, paragraph=False, width_ths=0.3)


def _to_items(results) -> list[dict]:
    items = []
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text:
            continue
        pts = np.array(bbox, dtype=float)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        items.append({
            "text": text, "conf": float(conf),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "h": max(y1 - y0, 1.0),
        })
    return items


def _group_lines(items: list[dict]) -> list[list[dict]]:
    """Gom ô chữ thành dòng với ngưỡng THÍCH ỨNG theo chiều cao chữ."""
    if not items:
        return []
    thr = max(8.0, 0.6 * float(np.median([i["h"] for i in items])))
    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_y = 0.0
    for it in sorted(items, key=lambda i: i["cy"]):
        if cur and abs(it["cy"] - cur_y) <= thr:
            cur.append(it)
            cur_y = float(np.mean([c["cy"] for c in cur]))   # trung bình trượt, không bị trôi
        else:
            if cur:
                lines.append(sorted(cur, key=lambda i: i["x0"]))
            cur, cur_y = [it], it["cy"]
    if cur:
        lines.append(sorted(cur, key=lambda i: i["x0"]))
    return lines


# ============================================================
# NHẬN DIỆN MÃ HỌC PHẦN
# ============================================================
_CODE_RE = re.compile(
    r"(?<![A-Z0-9])([A-Z]{2})\s?([0-9OQDILSBZ|]{4})(?:\s?[.,]\s?(\d))?(?![0-9A-Z])"
)
_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                            "|": "1", "S": "5", "B": "8", "Z": "2"})


def _find_code(text: str) -> tuple[str, str] | None:
    """Trả (mã gốc 6 ký tự, mã đầy đủ) hoặc None. Sửa lỗi O/0, I/1, S/5 trong phần số."""
    m = _CODE_RE.search(text.upper())
    if not m:
        return None
    letters, digits, suffix = m.group(1), m.group(2), m.group(3)
    if sum(c.isdigit() for c in digits) < 3:        # tránh nhầm từ thường thành mã
        return None
    base = letters + digits.translate(_DIGIT_FIX)
    return base, base + (f".{suffix}" if suffix else "")


_UNLEARNED_MARKERS = ("chua hoc", "chua co diem", "tong so tiet", "ma hoc phan", "tu chon")
_QT_TOKEN_RE = re.compile(r"^[QO0]T$", re.I)
_GRADE_LETTER_RE = re.compile(r"^[A-F](\s*\|\s*[A-F])?$|^BS$", re.I)


# ============================================================
# PHÂN TÍCH 1 TRANG
# ============================================================
def _parse_table_page(items: list[dict], state: dict) -> dict | None:
    """Chế độ chính: neo theo mã học phần. Trả None nếu trang không giống bảng điểm."""
    lines = _group_lines(items)

    # --- 1. Vị trí bắt đầu bảng "chưa học" (cắt bỏ phần này) ---
    cutoff_y = None
    for ln in lines:
        key = _norm_key(" ".join(i["text"] for i in ln))
        if any(m in key for m in _UNLEARNED_MARKERS[:3]):
            cutoff_y = min(i["y0"] for i in ln)
            break
    if cutoff_y is not None:
        state["in_unlearned"] = True
    elif state.get("in_unlearned"):
        cutoff_y = -1.0                              # cả trang thuộc bảng "chưa học"

    # --- 2. Cột TBCHP (chỉ dùng làm phương án dự phòng) ---
    for it in items:
        if re.fullmatch(r"tbc\s?hp", _norm_key(it["text"]).replace(" ", "")):
            state["tbchp_x"] = it["cx"]
            state["tbchp_w"] = max(it["x1"] - it["x0"], 25.0)

    # --- 3. Neo: mọi ô chứa mã học phần ---
    anchors = []
    anchor_src_ids = set()
    for it in items:
        found = _find_code(it["text"])
        if found:
            anchors.append({**it, "base": found[0], "full": found[1]})
            anchor_src_ids.add(id(it))
    anchors.sort(key=lambda a: a["cy"])
    if len(anchors) < 3 and not state.get("seen_codes"):
        return None
    if anchors:
        state["seen_codes"] = True

    out = {"rows": [], "ignored": []}
    if not anchors:
        return out

    gaps = np.diff([a["cy"] for a in anchors])
    med_gap = float(np.median(gaps)) if len(gaps) else 60.0
    max_dist = max(1.5 * med_gap, 40.0)

    # --- 4. Gán ô chữ vào hàng gần nhất theo Y ---
    rows = [{"anchor": a, "items": []} for a in anchors]
    ys = np.array([a["cy"] for a in anchors])
    for it in items:
        if id(it) in anchor_src_ids:
            continue
        k = int(np.argmin(np.abs(ys - it["cy"])))
        if abs(ys[k] - it["cy"]) <= max_dist:
            rows[k]["items"].append(it)

    pending: list[dict] = []
    for row in rows:
        a = row["anchor"]
        if cutoff_y is not None and a["cy"] >= cutoff_y:
            out["ignored"].append(a["full"])
            continue

        right = [i for i in row["items"] if i["cx"] > a["x1"] - 2]
        a_w = max(a["x1"] - a["x0"], 30.0)

        # ---- tên môn (chỉ để đối chiếu / dự phòng) ----
        name_items = [i for i in right
                      if i["cx"] < a["cx"] + 3.2 * a_w
                      and re.search(r"[^\W\d_]", i["text"])
                      and not _QT_TOKEN_RE.match(i["text"].strip())
                      and not _GRADE_LETTER_RE.match(i["text"].strip())]
        name_text = " ".join(" ".join(i["text"] for i in ln) for ln in _group_lines(name_items))

        # ---- xác định môn chuẩn: MÃ trước, TÊN sau ----
        subject = SUBJECT_CODES.get(a["base"])
        via = "code"
        by_name = match_subject(name_text) if name_text else None
        if subject is None and by_name is not None:
            subject, via = by_name, "name"
        elif subject is not None and by_name is not None and by_name != subject:
            state.setdefault("notes", []).append(
                f"Mã {a['full']} -> '{subject}' nhưng tên đọc được giống '{by_name}'. Đã ưu tiên theo mã."
            )
        if subject is None:
            out["ignored"].append(a["full"])
            continue

        # ---- dãy số theo thứ tự cột (trái -> phải) ----
        num_items = [i for i in right if i["cx"] > a["x1"] + 0.5 * a_w]
        num_items.sort(key=lambda i: (round(i["cx"] / 8), i["cy"]))
        toks: list[tuple] = []
        for i in num_items:
            nums = _numbers_with_pos(i["text"])
            for val, frac in nums:
                toks.append((val, i["x0"] + frac * (i["x1"] - i["x0"]), i["cx"], len(nums)))

        row_text = " ".join(i["text"] for i in sorted([a] + row["items"], key=lambda i: (round(i["cy"] / 10), i["x0"])))
        pending.append({"subject": subject, "via": via, "code": a["full"], "toks": toks,
                        "row_text": row_text, "row_items": row["items"],
                        "conf": min([i["conf"] for i in num_items] or [a["conf"]])})

    # --- 5. Tự hiệu chỉnh vị trí cột TBCHP từ các hàng đọc tốt (không phụ thuộc tiêu đề) ---
    page_w = max((i["x1"] for i in items), default=1000.0)
    tol = max(0.035 * page_w, 25.0)
    xs = [r[2] for p_ in pending for r in [pick_row_score(p_["toks"])] if r[1] == "pair" and r[2] is not None]
    if len(xs) >= 3:
        state["tbchp_x"] = float(np.median(xs))
    col_x = state.get("tbchp_x")
    out["tbchp_x"] = col_x

    # --- 6. Chọn điểm từng hàng, có kiểm tra cột ---
    for p_ in pending:
        score, method, _cx = pick_row_score(p_["toks"], col_x, tol, allow_column=False)

        # hàng của bảng "chưa học": chỉ có [tín chỉ, số tiết] (vd 3 | 60), không "QT", không có cặp điểm
        toks = p_["toks"]
        if (score is None and toks
                and any(t[0] <= 10 and float(t[0]).is_integer() for t in toks)
                and any(t[0] >= 30 and float(t[0]).is_integer() and t[0] % 15 == 0 for t in toks)
                and not re.search(r"\bQT\b", " ".join(i["text"] for i in p_["row_items"]), re.I)):
            out["ignored"].append(p_["code"])
            continue

        if score is None:                                   # dự phòng: đọc thẳng theo cột TBCHP
            score, method, _cx = pick_row_score(toks, col_x, tol, allow_column=True)

        dbg = "" if score is not None else f"  | số đọc được: {[t[0] for t in toks]}, cột TBCHP x≈{None if col_x is None else int(col_x)}"
        out["rows"].append({"subject": p_["subject"], "score": score, "code": p_["code"],
                            "via": p_["via"], "method": method, "ocr_text": p_["row_text"] + dbg,
                            "confidence": p_["conf"]})
    return out


def _parse_by_name(items: list[dict]) -> dict:
    """Chế độ dự phòng cho ảnh không có cột mã (vd bảng điểm giấy).

    Thử dòng của môn trước; chỉ nối thêm tối đa 2 dòng kế tiếp (không chứa môn khác)
    khi dòng đó chưa đủ để tìm cặp [TBCHP, hệ 4]. Tránh hút điểm của môn bên dưới.
    """
    lines = _group_lines(items)
    texts = [" ".join(i["text"] for i in ln) for ln in lines]
    matched = [match_subject(t) for t in texts]
    rows = []
    for idx, subj in enumerate(matched):
        if not subj:
            continue
        score, used = None, idx
        for k in range(3):
            j = idx + k
            if j >= len(lines) or (k > 0 and matched[j]):
                break
            score = pick_tbchp(_numbers_in(" ".join(texts[idx: j + 1])))
            used = j
            if score is not None:
                break
        if score is None:                                       # heuristic kiểu cũ, chỉ trên dòng gốc
            score, used = extract_score_from_text(texts[idx]), idx
        block = lines[idx: used + 1]
        rows.append({"subject": subj, "score": score, "code": None, "via": "name", "method": "text",
                     "ocr_text": " ".join(texts[idx: used + 1]),
                     "confidence": min(i["conf"] for ln in block for i in ln)})
    return {"rows": rows, "ignored": []}


# ============================================================
# API CHÍNH
# ============================================================
def _empty_result() -> dict:
    return {"scores": {}, "raw_text": [], "matched_lines": [], "unmatched_lines": [], "warnings": []}


def _finalize(res: dict, parsed: dict, state: dict) -> dict:
    for r in parsed["rows"]:
        if r["subject"] not in _STANDARD_SET:
            continue
        if r["score"] is None:
            res["unmatched_lines"].append(f"[{r['code'] or r['subject']}] không đọc được điểm: {r['ocr_text']}")
            continue
        prev = res["scores"].get(r["subject"])
        if prev is None or r["score"] > prev:                     # học lại: lấy điểm cao nhất
            res["scores"][r["subject"]] = r["score"]
        res["matched_lines"].append({k: r[k] for k in
                                     ("subject", "score", "ocr_text", "confidence", "code", "via", "method")})
    return res


def _add_warnings(res: dict, state: dict) -> dict:
    n = len(res["scores"])
    if n == 0:
        res["warnings"].append(
            "⚠️ Không trích xuất được điểm nào từ ảnh. Hãy kiểm tra: (1) ảnh có rõ nét không? "
            "(2) ảnh có chứa bảng điểm với mã học phần / tên môn và điểm số không?"
        )
    elif n < 5:
        res["warnings"].append(f"⚠️ Chỉ trích xuất được {n}/24 môn. Bạn nên kiểm tra lại các điểm đã trích xuất.")
    res["warnings"].extend("ℹ️ " + s for s in state.get("notes", []))
    res["not_found"] = [s for s in STANDARD_SUBJECTS if s not in res["scores"]]
    return res


def extract_scores_from_image(image_data, state: dict | None = None) -> dict:
    """Trích xuất điểm các môn chuẩn từ MỘT ảnh bảng điểm.

    Args:
        image_data: PIL Image, numpy array hoặc đường dẫn file ảnh.
        state: dict dùng chung giữa các trang của cùng 1 bảng điểm
               (nhớ vị trí cột TBCHP và việc đã vào bảng "chưa học").
               extract_scores_from_pdf() tự quản lý; ảnh đơn có thể bỏ trống.

    Returns:
        dict: scores, raw_text, matched_lines, unmatched_lines, warnings, not_found
    """
    state = state if state is not None else {}
    results = _run_ocr(image_data)
    items = _to_items(results)

    res = _empty_result()
    res["raw_text"] = [i["text"] for i in sorted(items, key=lambda i: (i["cy"], i["x0"]))]

    parsed = _parse_table_page(items, state)
    if parsed is None and not state.get("in_unlearned"):
        parsed = _parse_by_name(items)
    if parsed:
        _finalize(res, parsed, state)
    return _add_warnings(res, state)


def extract_scores_from_pdf(pdf_path, dpi: int = 300) -> dict:
    """PDF bảng điểm (kể cả PDF dạng ảnh scan / Print-to-PDF) -> điểm các môn chuẩn.

    Mỗi trang được render rồi OCR; trạng thái cột/bảng được truyền xuyên suốt các trang,
    nhờ đó trang 3, 4 (không có dòng tiêu đề) vẫn đọc đúng và bảng "chưa học" ở
    trang 4-6 bị bỏ qua hoàn toàn.
    """
    # pyrefly: ignore [missing-import]
    from pdf2image import convert_from_path

    state: dict = {}
    total = _empty_result()
    for page_no, page in enumerate(convert_from_path(str(pdf_path), dpi=dpi), 1):
        r = extract_scores_from_image(page, state=state)
        for subj, sc in r["scores"].items():
            if subj not in total["scores"] or sc > total["scores"][subj]:
                total["scores"][subj] = sc
        total["raw_text"] += r["raw_text"]
        total["matched_lines"] += [{**m, "page": page_no} for m in r["matched_lines"]]
        total["unmatched_lines"] += [f"(trang {page_no}) {u}" for u in r["unmatched_lines"]]
    return _add_warnings(total, state)


def build_full_score_dict(extracted_scores: dict[str, float]) -> dict[str, float]:
    """Đủ 24 môn; môn không có điểm (chưa học) = -1.0."""
    return {s: extracted_scores.get(s, -1.0) for s in STANDARD_SUBJECTS}