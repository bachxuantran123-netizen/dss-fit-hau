# Project Context — DSS FIT-HAU (Personal Decision Support System)

## 1. Project Overview
- **Hệ thống:** Ứng dụng Học máy cảnh báo và định hướng học tập cho sinh viên FIT-HAU.
- **Mục tiêu:** Cung cấp Hệ trợ giúp quyết định cá nhân (Personal DSS) phân tích điểm số nền tảng (Toán Rời Rạc, Lập Trình C, Cơ Sở Dữ Liệu) để gợi ý chuyên ngành (AI, SE, Cảnh báo học vụ).
- **Kiến trúc khép kín:** Phân tách hoàn toàn 2 môi trường:
  1. Offline Training (Huấn luyện ngoại tuyến, xuất model).
  2. Online Inference (Giao diện Web, nạp model để suy diễn).
- **Roadmap MVP:** Data Preprocessing → GridSearchCV Model Training → Streamlit Dashboard.
- **Ưu tiên cốt lõi:** Code sạch, dễ mở rộng, Explainable AI (XAI - Trực quan hóa luồng suy luận), và Bảo mật dữ liệu đầu vào (Validation).

## 2. Project Structure
- Python Monorepo Project.
- Cấu trúc thư mục chuẩn (Luôn tuân thủ khi tạo mới file):
  ```
  fit-hau-dss/
  ├── data/                              → Thư mục chứa dữ liệu
  │   ├── raw/                           → Data Kaggle thô (student-mat.csv)
  │   └── processed/                     → Data đã làm sạch (FIT_HAU_Cleaned.csv)
  ├── src/                               → Mã nguồn lõi
  │   ├── data_pipeline.py               → Xử lý dữ liệu & Labeling
  │   ├── train_core.py                  → Huấn luyện DecisionTree & Hyperparameter tuning
  │   └── app.py                         → Streamlit UI (Main App)
  ├── models/                            → Tri thức AI
  │   └── dss_brain.pkl                  → Model đã huấn luyện (Joblib)
  ├── reports/                           → Biểu đồ đánh giá
  │   ├── dss_confusion_matrix.png
  │   └── dss_tree.png
  ├── requirements.txt                   → Danh sách thư viện
  ├── .gitignore
  └── .antigravity/
      └── rules.md                      → AI Instructions (This file)
  ```

## 3. Coding Conventions & Standards
- **Stack:** Python 3.10+, Pandas, Scikit-learn, Streamlit, Plotly (cho Radar chart), Seaborn/Matplotlib, Openpyxl, Joblib. Không tự ý sử dụng thư viện Deep Learning (TensorFlow/PyTorch).
- **Naming Convention:**
- `snake_case` cho variables, methods, file names.
- `PascalCase` cho Classes.
- `UPPER_SNAKE_CASE` cho constants.
- **Type Hinting:** Bắt buộc sử dụng type hints cho parameters và return types (VD: `def process(data: pd.DataFrame) -> pd.DataFrame:`).
- **Virtual Environment:** Code được chạy trong môi trường `venv`. Không cấu hình cho Conda trừ khi user yêu cầu.

## 4. Architecture Patterns
- **Separation of Concerns (SoC):** Tuyệt đối KHÔNG viết logic `model.fit()` bên trong file `app.py`. UI chỉ làm nhiệm vụ giao tiếp và Inference.
- **Knowledge Transfer:** Huấn luyện model bằng `train_core.py`, dump ra `.pkl`. UI dùng `@st.cache_resource` để nạp `.pkl` vào RAM 1 lần duy nhất.
- **Zero-Trust Validation:** Mọi input từ người dùng trên giao diện phải bị chặn bằng `st.stop()` nếu giá trị `< 0` hoặc `> 10`.
- **Explainable AI (XAI):** Mọi quyết định của AI phải được dịch ra ngôn ngữ tự nhiên thông qua việc duyệt cấu trúc `decision_path` của Scikit-learn.

## 5. Response Style (AI Directives)
- Trả về code Python hoàn chỉnh, copy-paste được ngay.
- Không thêm dependency lạ ngoài stack đã định nghĩa.
- Không tự ý sửa file config trừ khi được yêu cầu
- Review code: dùng format có cấu trúc (bullet points, bảng), không viết đoạn văn dài
- Comment bằng tiếng Việt hoặc tiếng Anh tùy context, ưu tiên tiếng Anh cho code comment
- Giải thích lý do khi có nhiều cách tiếp cận, đề xuất phương án tốt nhất
- Nếu được yêu cầu tích hợp tính năng mới: Hãy xem xét kiến trúc hiện tại, đề xuất File cần sửa trước khi lao vào gen code.

## 6. Workflows & Modes
- **Task phức tạp** (feature mới, refactor nhiều file): đưa Implementation Plan trước, chờ xác nhận mới code
- **Task đơn giản** (fix 1 bug, sửa 1 method, thêm field): code ngay không cần plan
- **Trước khi sửa file**: đọc qua file hiện tại để hiểu context, không suy đoán nội dung
- **Data Engineering (`data_pipeline.py`):** Chỉ dùng Pandas vectorization. KHÔNG dùng vòng lặp `for` để duyệt dataset. Xử lý Missing values và ánh xạ cột rõ ràng.
- **AI Core (`train_core.py`):** Bắt buộc dùng `GridSearchCV` với cross-validation. Luôn set `random_state=42`. Bắt buộc sinh ảnh Confusion Matrix và Tree Plot.
- **Streamlit UI (`app.py`):** Dùng layout dạng Tabs (Trợ giúp & Phân tích). Áp dụng thiết kế Card-based. Tích hợp tính năng Download Excel qua `io.BytesIO()`.
- **Luôn kiểm tra lỗi:** Trước khi hoàn thành khối code, tự review xem có lỗi import, lỗi tham chiếu file model `.pkl`, hoặc quên `@st.cache_resource` hay không.

## 7. Module-specific Rules
### Data Engineering Module
- **Xử lý Missing Values:** Cần có báo cáo số dòng bị xóa/điền.
- **Feature Mapping:** Hard-code việc đổi tên G1, G2, G3 thành Toan_Roi_Rac, Lap_Trinh_C, Co_So_Du_Lieu.
- **Visualization:** Radar Chart cho kỹ năng cá nhân (Radar Plotly).
- **Caching:** Bắt buộc dùng `@st.cache_resource` khi load model để tránh tràn RAM server.
- **Export:** Sử dụng `io.BytesIO()` để tạo file Excel ảo trên RAM trước khi gọi nút Download, không lưu file Excel vật lý ra ổ cứng.

### Session Management
Cuối mỗi session, tự động tạo summary với format:
- Đang làm gì?
- Đã xong gì?
- Decision đã chốt
- Task tiếp theo