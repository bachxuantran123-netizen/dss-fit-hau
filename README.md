# 🎓 DSS FIT-HAU — Hệ Trợ Giúp Quyết Định Học Tập

> **Hệ thống cảnh báo và trợ giúp quyết định học tập cho sinh viên Khoa CNTT — Trường Đại học Kiến trúc Hà Nội (HAU)**

## 📋 Mô tả

Ứng dụng **Personal DSS (Decision Support System)** sử dụng Cây quyết định (Decision Tree) để:
- **Gợi ý chuyên ngành** phù hợp (AI, Software Engineering) dựa trên điểm 3 môn nền tảng
- **Cảnh báo học vụ** nếu điểm số không đạt ngưỡng
- **Giải thích minh bạch** lý do đề xuất (Explainable AI)

## 🏗️ Kiến trúc

```
Offline Training Pipeline          Online Inference (Web)
┌─────────────────────┐           ┌──────────────────────┐
│  data_pipeline.py   │           │      app.py          │
│  (Làm sạch & Nhãn)  │           │  (Streamlit UI)      │
│         ↓           │           │         ↑            │
│  train_core.py      │  ──.pkl──▶│  Load dss_brain.pkl  │
│  (GridSearchCV)     │           │  predict() + XAI     │
└─────────────────────┘           └──────────────────────┘
```

## 📁 Cấu trúc thư mục

```
DSS FIT-HAU/
├── data/
│   ├── raw/                → Dữ liệu thô từ Kaggle (student-mat.csv)
│   └── processed/          → Dữ liệu đã làm sạch (FIT_HAU_Cleaned.csv)
├── src/
│   ├── data_pipeline.py    → Xử lý dữ liệu & Gán nhãn
│   ├── train_core.py       → Huấn luyện Decision Tree (GridSearchCV)
│   └── app.py              → Giao diện Web Streamlit
├── models/                 → File mô hình .pkl (Joblib)
├── reports/                → Biểu đồ đánh giá (Confusion Matrix, Tree Plot)
├── documents/              → Tài liệu dự án (Backlog, báo cáo)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Cài đặt & Chạy

### 1. Clone repo
```bash
git clone https://github.com/<your-username>/dss-fit-hau.git
cd dss-fit-hau
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Cài thư viện
```bash
pip install -r requirements.txt
```

### 4. Chạy Pipeline (theo thứ tự)
```bash
# Sprint 1: Xử lý dữ liệu
python src/data_pipeline.py

# Sprint 2: Huấn luyện model
python src/train_core.py

# Sprint 3: Khởi động Web UI
streamlit run src/app.py
```

## 🗺️ Roadmap

| Sprint | Mục tiêu | Trạng thái |
|--------|----------|------------|
| Sprint 1 | Data Preprocessing & Labeling | 🔲 To Do |
| Sprint 2 | Model Training & Evaluation | 🔲 To Do |
| Sprint 3 | Streamlit Dashboard & Export | 🔲 To Do |

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — Xử lý dữ liệu
- **Scikit-learn** — Decision Tree + GridSearchCV
- **Streamlit** — Giao diện Web
- **Plotly** — Radar Chart
- **Seaborn / Matplotlib** — Confusion Matrix & Tree Plot
- **Openpyxl** — Xuất file Excel
- **Joblib** — Serialize model

## 👥 Nhóm phát triển

Sinh viên Khoa Công nghệ Thông tin — Trường Đại học Kiến trúc Hà Nội (FIT-HAU)
