# 🎓 DSS FIT-HAU — Hệ Trợ Giúp Quyết Định Học Tập

> **Hệ thống phân tích điểm số và định hướng nghề nghiệp cho sinh viên Khoa CNTT — Trường Đại học Kiến trúc Hà Nội (HAU)**

## 📋 Mô tả

Ứng dụng **Personal DSS (Decision Support System)** sử dụng kiến trúc **Hybrid 2 tầng + OCR** để:
- **Trích xuất điểm tự động từ ảnh** bảng điểm bằng AI-OCR (EasyOCR), hỗ trợ ảnh chụp bảng điểm giấy, screenshot portal trường, screenshot PDF.
- **Gán nhãn thông minh** bằng Content-Based Filtering (Cosine Similarity) giữa vector điểm sinh viên và Career Profile Vectors.
- **Dự đoán chuyên ngành** phù hợp nhất trong 5 hướng nghề nghiệp: Software Engineer, Data Engineer, AI Engineer, Security Engineer, System/DevOps.
- **Giải thích minh bạch** lý do đề xuất thông qua Explainable AI (XAI) bằng ngôn ngữ tự nhiên.
- **Trực quan hóa** điểm số bằng biểu đồ (Horizontal Bar Chart) và xuất báo cáo kết quả định dạng Excel.

## 🏗️ Kiến trúc Hệ thống (Hybrid 3 Tầng + OCR)

```
Offline Training Pipeline                                  Online Inference (Web)
┌────────────────────────────────────────┐                ┌─────────────────────────────────┐
│  1. pdf_extractor.py                   │                │         app.py                  │
│  (Cào điểm từ PDF)                     │                │    (Streamlit Wizard UI)        │
│         ↓                              │                │                                 │
│  2. data_pipeline.py                   │                │  ┌───────────────────────────┐  │
│  ┌─────────────────────────────────┐   │                │  │ OCR Layer (image_ocr.py)  │  │
│  │ Tầng 1: Content-Based Filtering │   │                │  │ Upload ảnh → EasyOCR      │  │
│  │ Cosine Sim(SV, Career Profile)  │   │                │  │ → Trích xuất điểm 24 môn  │  │
│  │ → Gán nhãn 5 chuyên ngành      │   │                │  └─────────────┬─────────────┘  │
│  └─────────────────────────────────┘   │                │                ↓                │
│         ↓                              │                │  Load dss_brain.pkl             │
│  3. train_core.py                      │ ── Xuất .pkl ──▶  predict() + XAI                │
│  ┌─────────────────────────────────┐   │                │                ↓                │
│  │ Tầng 2: Decision Tree           │   │                │  Tầng 3: Hybrid Score           │
│  │ GridSearchCV + Evaluation       │   │                │  (AI Score + Sở thích)          │
│  │ → Học từ nhãn, dự đoán + XAI    │   │                └─────────────────────────────────┘
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

### Tại sao Hybrid?

| Tầng | Phương pháp | Vai trò |
|------|-------------|---------|
| **OCR** | EasyOCR (Vietnamese + English) | Upload ảnh bảng điểm → AI tự động trích xuất tên môn & điểm số → Fuzzy keyword matching sang 24 môn chuẩn. |
| **Tầng 1** | Content-Based Filtering (Cosine Similarity) | Gán nhãn cho dữ liệu training — thay thế Skill Matrix heuristic cũ. So sánh pattern điểm 24 môn với 5 Career Profile Vectors. |
| **Tầng 2** | Decision Tree (GridSearchCV) | Học từ nhãn đã gán → dự đoán nhanh cho user mới + cung cấp giải thích (XAI) qua decision_path. |
| **Tầng 3** | Heuristic Rules (Hybrid Score) | Tích hợp online: Kết hợp xác suất từ AI với Trọng số Sở thích người dùng để đưa ra Bảng Xếp Hạng cá nhân hóa chính xác nhất. |

## 📁 Cấu trúc thư mục

```
DSS FIT-HAU/
├── data/
│   ├── raw/                → Data thô trích xuất từ bảng điểm PDF (FIT_HAU_Raw_Scores.csv)
│   ├── processed/          → Data đã làm sạch và gán nhãn (FIT_HAU_Cleaned.csv)
│   └── feedback.csv        → Dữ liệu đánh giá UAT của người dùng
├── src/
│   ├── pdf_extractor.py    → Trích xuất và gom bảng điểm từ hàng trăm file PDF
│   ├── merge_new_data.py   → Hợp nhất dữ liệu điểm 4 môn mới vào file điểm gốc
│   ├── data_pipeline.py    → Xử lý dữ liệu & Gán nhãn bằng Content-Based Filtering (Cosine Similarity)
│   ├── train_core.py       → Huấn luyện Decision Tree (GridSearchCV)
│   ├── image_ocr.py        → 🆕 Module OCR: Trích xuất điểm từ ảnh bảng điểm (EasyOCR)
│   └── app.py              → Giao diện Web Streamlit (Wizard Flow, OCR Upload, XAI, Hybrid Score)
├── models/                 → File mô hình .pkl (Joblib)
├── reports/                → Biểu đồ đánh giá (Confusion Matrix, Tree Plot)
├── documents/              → Tài liệu dự án (Backlog, kế hoạch, kiến trúc)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Hướng dẫn Cài đặt & Chạy

### 1. Clone repo
```bash
git clone https://github.com/bachxuantran123-netizen/dss-fit-hau.git
cd dss-fit-hau
```

### 2. Tạo môi trường ảo (Khuyến nghị)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Chạy Pipeline Hệ thống (Lần lượt)
```bash
# Bước 1: Trích xuất dữ liệu từ PDF (Tuỳ chọn nếu đã có file CSV thô)
python src/pdf_extractor.py

# Bước 1.5: Hợp nhất dữ liệu môn học mới (Tuỳ chọn nếu có file dữ liệu bổ sung)
python src/merge_new_data.py

# Bước 2: Làm sạch dữ liệu và Gán nhãn bằng Cosine Similarity (Tiền xử lý)
python src/data_pipeline.py

# Bước 3: Huấn luyện AI Model (Sinh ra file .pkl)
python src/train_core.py

# Bước 4: Khởi động Giao diện Web UI
python -m streamlit run src/app.py
```

### 5. Sử dụng tính năng OCR (Upload ảnh bảng điểm)

1. Truy cập Web UI → Nhập thông tin & sở thích (Step 1) → Bấm **"Tiếp theo"**
2. Tại Step 2, **upload ảnh bảng điểm** (PNG, JPG, JPEG, BMP, TIFF, WEBP)
3. Bấm **"🔍 Trích xuất điểm từ ảnh"** → AI-OCR tự động phân tích
4. Tại Step 2.5, **kiểm tra & chỉnh sửa** điểm đã trích xuất nếu cần
5. Bấm **"Hoàn tất & Xem Khuyến nghị"** → Xem kết quả AI

> **Lưu ý:** Lần đầu chạy OCR sẽ mất 1-2 phút để tải model EasyOCR (Vietnamese + English). Các lần sau sẽ nhanh hơn do model được cache.
>
> **Mẹo:** Ảnh nên rõ nét, không bị mờ/nghiêng, tên môn và điểm hiển thị đầy đủ để OCR đọc chính xác nhất.

## 🗺️ Roadmap & Tiến độ

| Sprint | Mục tiêu | Trạng thái |
|--------|----------|------------|
| Sprint 1 | Chuẩn bị dữ liệu (PDF Extractor, Data Cleaning, Heuristic Labeling) | ✅ Done |
| Sprint 2 | Huấn luyện mô hình (Decision Tree, GridSearchCV, Evaluation) | ✅ Done |
| Sprint 3 | Giao diện và Ứng dụng (Streamlit, Dynamic Form, XAI, Export Excel) | ✅ Done |
| Sprint 4 | Nâng cấp Bài cuối kỳ (Giao diện Wizard, Hybrid Score AI + Sở thích, Thu thập Feedback) | ✅ Done |
| Sprint 5 | 🆕 OCR Upload ảnh bảng điểm (EasyOCR, Fuzzy Matching, Review & Edit Flow) | ✅ Done |

## 🛠️ Tech Stack

- **Python 3.10+**
- **NumPy** — Tính toán vector & Cosine Similarity
- **pdfplumber** — Trích xuất dữ liệu bảng từ PDF
- **Pandas** — Tiền xử lý dữ liệu (Data Preprocessing)
- **Scikit-learn** — Machine Learning (Decision Tree + GridSearchCV)
- **Streamlit** — Phát triển Giao diện Web (Dashboard)
- **Plotly** — Trực quan hóa dữ liệu (Horizontal Bar Chart)
- **Seaborn / Matplotlib** — Vẽ Confusion Matrix & Tree Plot
- **Openpyxl** — Xuất báo cáo định dạng Excel
- **Joblib** — Đóng gói mô hình AI
- **EasyOCR** — 🆕 Trích xuất text từ ảnh bảng điểm (hỗ trợ tiếng Việt)
- **Pillow** — 🆕 Xử lý ảnh đầu vào (resize, convert format)

## 👥 Nhóm phát triển

Sinh viên Khoa Công nghệ Thông tin — Trường Đại học Kiến trúc Hà Nội (FIT-HAU)
