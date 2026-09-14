# Product Backlog - DSS Project

## Tóm tắt
- **Tổng số Task:** 15
- **Tổng số giờ ước tính:** 54 giờ
- **Các Sprint:** Sprint 1, Sprint 2, Sprint 3

## Chi tiết Backlog

| Sprint   | Epic                          | User Story                              | Task                                                                               | Người phụ trách   | Status   | Priority   |   Estimate (Hours) | Sản phẩm bàn giao (Deliverable)            |
|:---------|:------------------------------|:----------------------------------------|:-----------------------------------------------------------------------------------|:------------------|:---------|:-----------|-------------------:|:-------------------------------------------|
| Sprint 1 | Epic 1: Chuẩn bị Dữ liệu      | US 1.1: Quản lý và làm sạch dữ liệu     | Task 1: Tải dataset Student Performance từ Kaggle và nạp bằng Pandas               | Dev 1 (Data)      | Done     | High       |                  2 | Script nạp dữ liệu - file dataset.csv      |
| Sprint 1 | Epic 1: Chuẩn bị Dữ liệu      | US 1.1: Quản lý và làm sạch dữ liệu     | Task 2: Xử lý dữ liệu khuyết (Missing Values) và xóa cột không liên quan           | Dev 1 (Data)      | Done     | High       |                  3 | DataFrame đã làm sạch (Cleaned data)       |
| Sprint 1 | Epic 1: Chuẩn bị Dữ liệu      | US 1.2: Chuẩn hóa đặc trưng và Gán nhãn | Task 3: Đổi tên cột thành Toán Rời Rạc; Lập Trình C; Cơ Sở Dữ Liệu                 | Dev 2 (Data)      | Done     | Medium     |                  2 | Dict mapping và cột dữ liệu chuẩn hóa      |
| Sprint 1 | Epic 1: Chuẩn bị Dữ liệu      | US 1.2: Chuẩn hóa đặc trưng và Gán nhãn | Task 4: Quy đổi thang 20 về thang 10 và gán nhãn 3 chuyên ngành (AI; SE; Cảnh báo) | Dev 2 (Data)      | Done     | High       |                  4 | Cột nhãn phân loại hoàn chỉnh              |
| Sprint 2 | Epic 2: Huấn luyện Mô hình AI | US 2.1: Huấn luyện và Tối ưu mô hình    | Task 1: Viết script chia tập Train/Test (80-20)                                    | Dev 3 (AI Core)   | Done     | High       |                  2 | Module chia dữ liệu X_train - y_train      |
| Sprint 2 | Epic 2: Huấn luyện Mô hình AI | US 2.1: Huấn luyện và Tối ưu mô hình    | Task 2: Tích hợp GridSearchCV để tìm max_depth tối ưu cho Decision Tree            | Dev 3 (AI Core)   | Done     | Medium     |                  6 | Báo cáo tham số tối ưu (best parameters)   |
| Sprint 2 | Epic 2: Huấn luyện Mô hình AI | US 2.2: Đánh giá và Đóng gói            | Task 3: Đánh giá Accuracy và lưu ảnh Ma trận nhầm lẫn (dss_confusion_matrix.png)   | Dev 4 (AI/UI)     | Done     | High       |                  3 | File ảnh dss_confusion_matrix.png          |
| Sprint 2 | Epic 2: Huấn luyện Mô hình AI | US 2.2: Đánh giá và Đóng gói            | Task 4: Vẽ và lưu sơ đồ kiến trúc Cây Quyết Định (dss_tree.png)                    | Dev 4 (AI/UI)     | Done     | Medium     |                  4 | File sơ đồ dss_tree.png                    |
| Sprint 2 | Epic 2: Huấn luyện Mô hình AI | US 2.2: Đánh giá và Đóng gói            | Task 5: Dùng Joblib đóng gói mô hình tốt nhất thành file dss_brain.pkl             | Dev 3 (AI Core)   | Done     | High       |                  2 | Model artifact dss_brain.pkl               |
| Sprint 3 | Epic 3: Giao diện và Ứng dụng | US 3.1: Xây dựng Giao diện người dùng   | Task 1: Khởi tạo Streamlit App với 2 Tabs (Trợ giúp quyết định & Phân tích)        | Dev 5 (Frontend)  | To Do    | High       |                  3 | Giao diện cơ sở trên app.py                |
| Sprint 3 | Epic 3: Giao diện và Ứng dụng | US 3.1: Xây dựng Giao diện người dùng   | Task 2: Viết rào chắn chặn điểm âm (<0) hoặc quá lớn (>10)                         | Dev 5 (Frontend)  | To Do    | Medium     |                  2 | Form validation & cảnh báo lỗi người dùng  |
| Sprint 3 | Epic 3: Giao diện và Ứng dụng | US 3.2: Tích hợp tính năng cốt lõi      | Task 3: Viết hàm decision_path để nội soi cây quyết định và dịch ra tiếng Việt     | Dev 3 (AI Core)   | To Do    | High       |                  7 | Chuỗi luật suy diễn logic hiển thị trên UI |
| Sprint 3 | Epic 3: Giao diện và Ứng dụng | US 3.2: Tích hợp tính năng cốt lõi      | Task 4: Hiển thị ảnh Ma trận và Cây máy học lên Tab 2 của Streamlit                | Dev 1 (Data)      | To Do    | High       |                  3 | Tab Phân tích tích hợp hoàn chỉnh hình ảnh |
| Sprint 3 | Epic 4: Báo cáo và Kiểm thử   | US 4.1: Xuất báo cáo                    | Task 1: Đóng gói dữ liệu đầu vào & kết quả thành phiếu Excel (Openpyxl)            | Dev 4 (AI/UI)     | To Do    | Medium     |                  5 | Tính năng xuất file .xlsx định dạng chuẩn  |
| Sprint 3 | Epic 4: Báo cáo và Kiểm thử   | US 4.2: Đảm bảo chất lượng (QA)         | Task 2: Kiểm thử End-to-End toàn bộ hệ thống từ nhập điểm đến tải Excel            | Cả nhóm           | To Do    | High       |                  6 | Test cases & biên bản nghiệm thu không lỗi |