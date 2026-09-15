# 🎓 DSS FIT-HAU — Hệ Trợ Giúp Quyết Định Học Tập

> **Hệ thống phân tích điểm số và định hướng nghề nghiệp cho sinh viên Khoa CNTT — Trường Đại học Kiến trúc Hà Nội (HAU)**

## 📋 Mô tả

Ứng dụng **Personal DSS (Decision Support System)** sử dụng kiến trúc **Hybrid 2 tầng** để:
- **Gán nhãn thông minh** bằng Content-Based Filtering (Cosine Similarity) giữa vector điểm sinh viên và Career Profile Vectors.
- **Dự đoán chuyên ngành** phù hợp nhất trong 5 hướng nghề nghiệp: Software Engineer, Data Engineer, AI Engineer, Security Engineer, System/DevOps.
- **Giải thích minh bạch** lý do đề xuất thông qua Explainable AI (XAI) bằng ngôn ngữ tự nhiên.
- **Trực quan hóa** điểm số bằng biểu đồ (Horizontal Bar Chart) và xuất báo cáo kết quả định dạng Excel.

## 🏗️ Kiến trúc Hệ thống (Hybrid 2 Tầng)

```
Offline Training Pipeline                                  Online Inference (Web)
┌────────────────────────────────────────┐                ┌──────────────────────┐
│  1. pdf_extractor.py                   │                │      app.py          │
│  (Cào điểm từ PDF)                     │                │  (Streamlit UI)      │
│         ↓                              │                │         ↑            │
│  2. data_pipeline.py                   │                │  Load dss_brain.pkl  │
│  ┌─────────────────────────────────┐   │                │  predict() + XAI     │
│  │ Tầng 1: Content-Based Filtering │   │                └──────────────────────┘
│  │ Cosine Sim(SV, Career Profile)  │   │
│  │ → Gán nhãn 5 chuyên ngành      │   │
│  └─────────────────────────────────┘   │
│         ↓                              │
│  3. train_core.py                      │ ── Xuất .pkl ──▶
│  ┌─────────────────────────────────┐   │
│  │ Tầng 2: Decision Tree           │   │
│  │ GridSearchCV + Evaluation        │   │
│  │ → Học từ nhãn, dự đoán + XAI   │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

### Tại sao Hybrid?

| Tầng | Phương pháp | Vai trò |
|------|-------------|---------|
| **Tầng 1** | Content-Based Filtering (Cosine Similarity) | Gán nhãn cho dữ liệu training — thay thế Skill Matrix heuristic cũ. So sánh pattern điểm 20 môn với 5 Career Profile Vectors. |
| **Tầng 2** | Decision Tree (GridSearchCV) | Học từ nhãn đã gán → dự đoán nhanh cho user mới + cung cấp giải thích (XAI) qua decision_path. |

## 📁 Cấu trúc thư mục

```
DSS FIT-HAU/
├── data/
│   ├── raw/                → Data thô trích xuất từ bảng điểm PDF (FIT_HAU_Raw_Scores.csv)
│   └── processed/          → Data đã làm sạch và gán nhãn (FIT_HAU_Cleaned.csv)
├── src/
│   ├── pdf_extractor.py    → Trích xuất và gom bảng điểm từ hàng trăm file PDF
│   ├── data_pipeline.py    → Xử lý dữ liệu & Gán nhãn bằng Content-Based Filtering (Cosine Similarity)
│   ├── train_core.py       → Huấn luyện Decision Tree (GridSearchCV)
│   └── app.py              → Giao diện Web Streamlit (Dynamic Form + XAI)
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

# Bước 2: Làm sạch dữ liệu và Gán nhãn bằng Cosine Similarity (Tiền xử lý)
python src/data_pipeline.py

# Bước 3: Huấn luyện AI Model (Sinh ra file .pkl)
python src/train_core.py

# Bước 4: Khởi động Giao diện Web UI
streamlit run src/app.py
```

## 🗺️ Roadmap & Tiến độ

| Sprint | Mục tiêu | Trạng thái |
|--------|----------|------------|
| Sprint 1 | Chuẩn bị dữ liệu (PDF Extractor, Data Cleaning, Heuristic Labeling) | ✅ Done |
| Sprint 2 | Huấn luyện mô hình (Decision Tree, GridSearchCV, Evaluation) | ✅ Done |
| Sprint 3 | Giao diện và Ứng dụng (Streamlit, Dynamic Form, XAI, Export Excel) | ✅ Done |
| Sprint 4 | Nâng cấp kiến trúc Hybrid (Content-Based Filtering + Decision Tree) | ✅ Done |

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

## 👥 Nhóm phát triển

Sinh viên Khoa Công nghệ Thông tin — Trường Đại học Kiến trúc Hà Nội (FIT-HAU)
