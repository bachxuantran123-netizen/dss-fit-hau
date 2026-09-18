# 📘 Hướng dẫn Trả lời Vấn đáp: Lý thuyết Thuật toán DSS FIT-HAU

Tài liệu này tổng hợp **lý thuyết cốt lõi**, **vị trí áp dụng trong code**, và **lý do lựa chọn (giải thích bảo vệ)** dành cho 3 thuật toán/phương pháp chính cấu thành nên Kiến trúc Hybrid 3 tầng của hệ thống.

---

## 1. Tầng 1: Content-Based Filtering & Cosine Similarity
*(Tiền xử lý và Gán nhãn dữ liệu - Tự động hóa gán nhãn)*

### a. Lý thuyết
**Cosine Similarity (Độ tương đồng Cosine)** là thước đo toán học dùng để đánh giá mức độ giống nhau giữa hai vector trong không gian đa chiều (n-dimensional space). 
- Thay vì đo khoảng cách hình học (khoảng cách Euclidean), nó đo **góc ($\theta$)** giữa hai vector.
- **Công thức:** $cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$ (Tích vô hướng của 2 vector chia cho tích độ dài của chúng).
- **Kết quả:** Trả về giá trị từ 0 đến 1 (nếu các thành phần dương). Giá trị càng gần 1, hai vector càng giống nhau về mặt xu hướng (pattern).

### b. Áp dụng ở đâu trong project?
- **File:** `src/data_pipeline.py`
- **Cách hoạt động:** Bảng điểm 20 môn của một sinh viên được coi là một vector A (20 chiều). Hệ thống định nghĩa trước 5 "Career Profile Vectors" (vector chuyên ngành lý tưởng, ví dụ Data Engineer thì điểm Toán, CSDL, Python mặc định là 10).
- Hệ thống tính Cosine Similarity giữa vector bảng điểm của sinh viên với 5 vector lý tưởng này. Độ tương đồng với vector nào lớn nhất thì sinh viên đó sẽ được **gán nhãn (Label)** chuyên ngành đó.

### c. Câu hỏi phòng vệ: "Tại sao không dùng luật if-else (Heuristic) để gán nhãn cho nhanh?"
> **Trả lời:** "Nếu dùng if-else (ví dụ: `if diem_Toan > 8 and diem_CSDL > 8 -> Data`), các quy tắc sẽ rất cứng nhắc và không phủ hết được các trường hợp sinh viên học lệch (ví dụ Toán 10 nhưng CSDL chỉ 6). Cosine Similarity đánh giá được **toàn cục** xu hướng (pattern) của cả 20 môn học so với hình mẫu chuyên ngành, do đó việc gán nhãn tập dữ liệu huấn luyện (Training Set) mang tính mềm dẻo, chính xác và có cơ sở toán học vững chắc hơn."

---

## 2. Tầng 2: Thuật toán Decision Tree (Cây Quyết Định)
*(Core AI - Học máy có giám sát & Giải thích quyết định)*

### a. Lý thuyết
**Decision Tree (CART - Classification and Regression Trees)** là thuật toán học máy có giám sát (Supervised Learning). 
- Thuật toán chia cắt không gian dữ liệu bằng cách đặt ra các câu hỏi dạng điều kiện (ví dụ: `Điểm Python <= 7.5?`) tại các nút (node).
- Nó chọn đặc trưng (môn học) tốt nhất để chia cắt dựa trên việc tối đa hóa **Information Gain** (độ lợi thông tin) hoặc giảm thiểu **Gini Impurity** (độ vẩn đục/lẫn lộn của dữ liệu). Quá trình này lặp lại cho đến khi tạo thành các lá (leaf) chứa kết quả phân loại cuối cùng.

### b. Áp dụng ở đâu trong project?
- **File Huấn luyện:** `src/train_core.py` (Huấn luyện từ dữ liệu đã gán nhãn ở Tầng 1. Áp dụng thêm `GridSearchCV` để tìm siêu tham số tối ưu như `max_depth` nhằm tránh Overfitting).
- **File Ứng dụng (Web):** `src/app.py` (Sử dụng hàm `predict_proba` để đưa ra xác suất 5 ngành).
- **Điểm nhấn XAI (Explainable AI):** Dùng phương thức `decision_path` của thư viện Scikit-Learn trong `app.py` để lần ngược các node trên cây, từ đó trích xuất ra các luật logic. Hiển thị lên UI bằng ngôn ngữ tự nhiên: *"Tại sao chọn Software? Vì điểm OOP > 7.5 và điểm CTDL > 7.0"*.

### c. Câu hỏi phòng vệ: "Tại sao dùng Decision Tree mà không dùng thuật toán mạnh hơn như Random Forest, SVM hay Mạng Nơ-ron (Deep Learning)?"
> **Trả lời:** "Bản chất của một Hệ Trợ Giúp Quyết Định (Decision Support System - DSS) trong môi trường giáo dục không chỉ là đưa ra kết quả, mà phải **giải thích được (Explainability)** cho sinh viên hiểu TẠI SAO hệ thống lại khuyên như vậy (White-box model). Nếu dùng Neural Network (Black-box), độ chính xác có thể tăng 1-2% nhưng không thể giải thích được luồng suy luận. Decision Tree cho phép trích xuất bộ luật (Rule-based) dễ dàng bằng `decision_path`, giúp hệ thống trở nên minh bạch và thuyết phục hơn với người dùng. Ngoài ra dữ liệu dạng điểm số dạng bảng (Tabular Data) rất phù hợp với Decision Tree."

---

## 3. Tầng 3: Hybrid Score (Hệ chuyên gia kết hợp trọng số)
*(Cá nhân hóa và Tối ưu trải nghiệm - Recommendation)*

### a. Lý thuyết
Đây là một biến thể của **Hệ chuyên gia dựa trên luật (Rule-based System)** và **Đánh giá đa tiêu chí (Multi-criteria scoring)**. Việc ra quyết định không chỉ phụ thuộc vào một mô hình duy nhất (AI), mà kết hợp với các rào cản/luật logic ngoại sinh (những tham số ngoài thuật toán) để điều chỉnh đầu ra.

### b. Áp dụng ở đâu trong project?
- **File:** `src/app.py`
- **Cách hoạt động:** AI Decision Tree (Tầng 2) trả về một mảng xác suất (Probabilities), ví dụ: `[Software: 40%, Data: 35%, AI: 25%]`.
- Hệ thống xét form **Sở thích** của người dùng ở Màn hình 1. Nếu người dùng tích chọn "Thích làm việc với dữ liệu", thuật toán sẽ cộng thêm một lượng **Bonus Trọng số (10% tức 0.1)** vào điểm xác suất của Data Engineer.
- Sau đó, chuẩn hóa lại (Normalize) để tổng bằng 100% và tiến hành xếp hạng (Rank Top 1, Top 2, Top 3).

### c. Câu hỏi phòng vệ: "Việc cộng thêm điểm sở thích có làm sai lệch độ chính xác của AI không?"
> **Trả lời:** "Nó không làm sai lệch, mà ngược lại khắc phục được **nhược điểm chí mạng của AI thuần túy**. AI chỉ nhìn vào dữ liệu điểm số quá khứ. Ví dụ, sinh viên A và B có điểm số giống hệt nhau (do cùng chăm chỉ học), AI sẽ khuyên cả hai học cùng 1 chuyên ngành. Nhưng nếu A thích Code Web, B thích An toàn thông tin, thì kết quả phải khác nhau. Thuật toán Hybrid Score đảm bảo AI đóng vai trò **cơ sở năng lực (chiếm 90%)**, còn Sở thích đóng vai trò **chất xúc tác (chiếm 10%)** để bẻ lái kết quả phù hợp với đam mê cá nhân. Đây chính là tiêu chuẩn của một Hệ tư vấn cá nhân hóa (Personalized Recommender System) hiện đại."
