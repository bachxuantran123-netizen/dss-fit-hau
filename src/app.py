"""
app.py — Streamlit Dashboard (Main Entry Point)
============================================================
Giao diện Web cho Hệ Trợ Giúp Quyết Định Học Tập FIT-HAU.

Architecture (Hybrid 2 tầng + Rule-based Preferences):
    - Tầng 1: Content-Based Filtering (Cosine Similarity) → Gán nhãn
    - Tầng 2: Decision Tree → Dự đoán + XAI
    - Tầng 3 (Mới): Hybrid Score = AI Score + Preference Bonus
    - OCR (Mới): Upload ảnh bảng điểm → Tự động trích xuất điểm
"""

import os
import io
import csv
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from PIL import Image

# Local OCR module
from image_ocr import extract_scores_from_image, build_full_score_dict, STANDARD_SUBJECTS
from pdf_extractor import extract_scores_from_student_pdf

# ============================================================
# CONSTANTS & CONFIGS
# ============================================================
MODEL_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")
FEEDBACK_PATH: str = os.path.join("data", "feedback.csv")

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0
PREFERENCE_BONUS: float = 0.10  # Bonus cộng thêm cho mỗi sở thích phù hợp

# Ánh xạ sở thích sang chuyên ngành
PREFERENCES_MAP = {
    "Thích viết code, tạo ra phần mềm/ứng dụng (Frontend, Backend)": "Software Engineer",
    "Thích làm việc với dữ liệu, truy vấn SQL, thiết kế CSDL": "Data Engineer",
    "Thích toán học, thuật toán, máy học, thị giác máy tính": "AI Engineer",
    "Thích tìm hiểu về hacker, mã hóa, an toàn thông tin mạng": "Security Engineer",
    "Thích cài đặt máy chủ, Linux, hạ tầng mạng, Cloud": "System/DevOps"
}

# Khuyến nghị hành động
CAREER_ADVICE = {
    "Software Engineer": "💡 **Khuyến nghị**: Bạn có tư duy logic và kỹ năng lập trình tốt. Hãy tập trung làm chủ 1-2 ngôn ngữ cốt lõi (Java, C#, Python), tìm hiểu sâu về Framework (Spring Boot, React/Vue) và tự làm các dự án thực tế để xây dựng Portfolio trên GitHub.",
    "Data Engineer": "💡 **Khuyến nghị**: Bạn có thế mạnh lớn về tổ chức dữ liệu. Hãy học thêm về SQL nâng cao, Python (thư viện Pandas), các công cụ Big Data (Hadoop, Spark) và kiến trúc kho dữ liệu (Data Warehouse).",
    "AI Engineer": "💡 **Khuyến nghị**: Khả năng Toán và Thuật toán của bạn rất tuyệt vời. Hãy đào sâu vào Machine Learning, Deep Learning (TensorFlow/PyTorch) và làm quen với bài toán xử lý ngôn ngữ tự nhiên (NLP) hoặc thị giác máy tính (CV).",
    "Security Engineer": "💡 **Khuyến nghị**: Sự cẩn thận và kiến thức mạng là điểm mạnh của bạn. Nên hướng tới việc lấy các chứng chỉ quốc tế (CEH, CompTIA Security+), thực hành thường xuyên trên TryHackMe/HackTheBox và học sâu về mã hóa.",
    "System/DevOps": "💡 **Khuyến nghị**: Bạn am hiểu kiến trúc hệ thống rất tốt. Hãy tìm hiểu thêm về hệ điều hành Linux, công nghệ container hóa (Docker, Kubernetes), CI/CD pipelines (Jenkins, GitLab CI) và các dịch vụ Cloud (AWS, Azure)."
}

# ============================================================
# CORE FUNCTIONS
# ============================================================
@st.cache_resource
def load_model():
    """Load trained Decision Tree model from .pkl file."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Lỗi: Không tìm thấy file mô hình tại `{MODEL_PATH}`. Vui lòng huấn luyện mô hình trước.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
        st.stop()


def validate_scores(scores: dict[str, float]) -> tuple[bool, str]:
    """Zero-Trust validation: kiểm tra điểm hợp lệ.
    
    Giá trị hợp lệ:
        -1.0 = Chưa học (bỏ qua)
        0.0 - 10.0 = Điểm thực tế
    """
    for subject, score in scores.items():
        if score == -1.0:
            continue  # -1.0 = chưa học, hợp lệ
        if score < MIN_SCORE or score > MAX_SCORE:
            return False, f"🚨 Điểm số không hợp lệ ở môn **{subject}**: `{score}`. Vui lòng nhập điểm từ {MIN_SCORE} đến {MAX_SCORE}."
    return True, ""


def explain_decision(model, input_data: pd.DataFrame) -> list[dict]:
    """Trace the decision_path of the Decision Tree and extract logic."""
    node_indicator = model.decision_path(input_data)
    leaf_id = model.apply(input_data)
    
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    feature_names = model.feature_names_in_
    
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]
    
    explanation_steps = []
    
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue
            
        feature_idx = feature[node_id]
        if feature_idx >= len(feature_names) or feature_idx < 0:
            continue
            
        subject_name = feature_names[feature_idx]
        feature_val = input_data.iloc[sample_id, feature_idx]
        thres = threshold[node_id]
        
        condition = "<=" if feature_val <= thres else ">"
        explanation_steps.append({
            "subject": subject_name,
            "score": feature_val,
            "operator": condition,
            "threshold": thres
        })
            
    return explanation_steps


def create_bar_chart(scores: dict[str, float]) -> go.Figure:
    """Create a Horizontal Bar Chart visualizing student scores."""
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    subjects = list(sorted_scores.keys())
    values = list(sorted_scores.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=subjects,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"{v:.1f}" for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Biểu đồ Năng lực Học tập",
        xaxis_title="Điểm số",
        xaxis=dict(range=[0, 10]),
        height=max(400, len(subjects) * 30),
        margin=dict(l=200, r=20, t=40, b=20)
    )
    return fig


def generate_excel_report(user_name: str, student_data: dict, top1_prediction: str, explanation_steps: list[dict]) -> bytes:
    """Generate an Excel report in-memory using io.BytesIO()."""
    df = pd.DataFrame([student_data])
    df.insert(0, "Tên Sinh Viên", user_name if user_name else "Ẩn danh")
    df["Dự đoán chuyên ngành"] = top1_prediction
    
    explanation_str = " ➔ ".join([
        f"Vì {s['subject']} ({s['score']:.1f}) {s['operator']} {s['threshold']:.2f}" 
        for s in explanation_steps
    ]) if explanation_steps else "Không có giải thích"
    
    df["Giải thích (XAI)"] = explanation_str
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Ket_Qua_Tu_Van')
    
    processed_data = output.getvalue()
    return processed_data


# ============================================================
# SYSTEM EVALUATION (FEEDBACK LOOP)
# ============================================================
def display_system_metrics():
    """Hiển thị điểm đánh giá trung bình trên Sidebar."""
    if os.path.exists(FEEDBACK_PATH):
        try:
            df_fb = pd.read_csv(FEEDBACK_PATH, encoding='utf-8-sig')
            if not df_fb.empty:
                avg_rating = df_fb['Rating'].mean()
                total = len(df_fb)
                st.metric(
                    label="🌟 Đánh giá hệ thống", 
                    value=f"{avg_rating:.1f} ⭐", 
                    delta=f"{total} lượt đánh giá"
                )
        except Exception:
            pass

def save_feedback(name: str, suggested_career: str, rating: int, comment: str):
    """Lưu đánh giá của người dùng vào file CSV."""
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    file_exists = os.path.exists(FEEDBACK_PATH)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_name = name if name else "Ẩn danh"
    safe_comment = comment if comment else "Không có"
    
    with open(FEEDBACK_PATH, "a", encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Name", "Suggested_Career", "Rating", "Comment"])
        writer.writerow([timestamp, safe_name, suggested_career, rating, safe_comment])


# ============================================================
# MAIN APP
# ============================================================
def main() -> None:
    st.set_page_config(
        page_title="DSS FIT-HAU — Trợ Giúp Quyết Định",
        page_icon="🎓",
        layout="wide",
    )

    model = load_model()
    if not hasattr(model, 'feature_names_in_'):
        st.error("⚠️ Mô hình quá cũ. Vui lòng chạy `train_core.py`.")
        st.stop()
    features = model.feature_names_in_

    # --- Sidebar ---
    with st.sidebar:
        st.title("🎓 DSS FIT-HAU")
        st.markdown("**Hệ Trợ Giúp Quyết Định Học Tập**")
        st.divider()
        st.caption("Powered by: ML Decision Tree & Rule-based Engine")
        
        # Gọi hàm hiển thị metric tự động cập nhật
        display_system_metrics()

    # --- Main Content ---
    st.title("🎓 Hệ Thống Trợ Giúp Quyết Định (DSS)")
    st.markdown("Khám phá chuyên ngành phù hợp nhất với năng lực và sở thích của bạn.")

    tab1, tab2 = st.tabs(["🧭 Trợ Giúp Lựa Chọn (Tư vấn)", "📊 Phân Tích Mô Hình AI"])

    with tab1:
        # Initialize session state for Wizard flow
        if 'step' not in st.session_state:
            st.session_state.step = 1
            
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
            
        if 'user_id' not in st.session_state:
            st.session_state.user_id = ""
            
        if 'selected_prefs' not in st.session_state:
            st.session_state.selected_prefs = []

        # --- STEP 1: INFO & PREFERENCES ---
        if st.session_state.step == 1:
            st.subheader("👤 Thông tin cá nhân & Sở thích")
            st.markdown("Hãy chia sẻ một chút về bạn để hệ thống có thể đưa ra gợi ý cá nhân hóa tốt nhất.")
            
            col_info, col_pref = st.columns(2)
            with col_info:
                st.markdown("#### Thông tin")
                is_anonymous = st.checkbox("🕵️‍♂️ Chế độ Ẩn danh (Không nhập tên/mã SV)", value=st.session_state.get('is_anonymous', False))
                
                if not is_anonymous:
                    user_name = st.text_input("Họ và Tên", value=st.session_state.user_name)
                    user_id = st.text_input("Mã sinh viên", value=st.session_state.user_id)
                else:
                    user_name = "Ẩn danh"
                    user_id = ""
                    st.info("Chế độ ẩn danh được bật. Bạn có thể tiếp tục mà không cần để lại thông tin.")
            
            with col_pref:
                st.markdown("#### Sở thích (*Bắt buộc*)")
                st.caption("Đánh dấu vào những công việc bạn thích làm nhất:")
                selected_prefs = []
                for pref_text, career in PREFERENCES_MAP.items():
                    if st.checkbox(pref_text, value=(career in st.session_state.selected_prefs)):
                        selected_prefs.append(career)
                
                st.markdown("---")
                is_undecided = st.checkbox("🤔 Chưa xác định được sở thích riêng", value=st.session_state.get('is_undecided', False))
                        
            st.divider()
            
            # Form Validation cho Sở thích
            if not selected_prefs and not is_undecided:
                st.warning("⚠️ Vui lòng chọn ít nhất một sở thích ở trên, hoặc tích vào ô 'Chưa xác định được sở thích riêng' để tiếp tục.")
                
            if st.button("Tiếp theo", type="primary", disabled=(not selected_prefs and not is_undecided)):
                if is_undecided:
                    selected_prefs = []  # Xóa sở thích nếu người dùng báo là chưa xác định
                    
                st.session_state.is_anonymous = is_anonymous
                st.session_state.user_name = user_name
                st.session_state.user_id = user_id
                st.session_state.selected_prefs = selected_prefs
                st.session_state.is_undecided = is_undecided
                st.session_state.step = 2
                st.rerun()

        # --- STEP 2: UPLOAD ẢNH BẢNG ĐIỂM (Hỗ trợ nhiều ảnh) ---
        elif st.session_state.step == 2:
            st.subheader("📸 Upload ảnh bảng điểm")
            greeting_name = st.session_state.user_name if st.session_state.user_name else "bạn"
            st.info(
                f"Chào **{greeting_name}**, vui lòng tải lên ảnh bảng điểm của bạn. "
                "Hệ thống sẽ tự động đọc và trích xuất điểm từ ảnh.\n\n"
                "**Hỗ trợ:** Ảnh chụp bảng điểm giấy, screenshot portal trường, screenshot file PDF.\n\n"
                "💡 **Có thể upload 1 hoặc nhiều ảnh** — nếu bảng điểm dài, hãy chụp nhiều ảnh rồi chọn tất cả cùng lúc."
            )
            
            # File uploader — hỗ trợ nhiều ảnh/pdf
            uploaded_files = st.file_uploader(
                "Chọn file/ảnh bảng điểm (có thể chọn nhiều file)",
                type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                help="Chấp nhận các định dạng: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP. Giữ Ctrl để chọn nhiều file.",
                key="image_uploader",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Hiển thị tất cả ảnh đã upload
                st.markdown(f"**📂 Đã tải lên {len(uploaded_files)} ảnh:**")
                
                cols_preview = st.columns(min(len(uploaded_files), 3))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols_preview[idx % 3]:
                        if uploaded_file.name.lower().endswith(".pdf"):
                            st.info(f"📄 PDF {idx + 1}: {uploaded_file.name}")
                        else:
                            img = Image.open(uploaded_file)
                            st.image(img, caption=f"🖼️ Ảnh {idx + 1}: {uploaded_file.name}", use_container_width=True)
                
                # Nút bấm để bắt đầu trích xuất
                if st.button("🔍 Trích xuất điểm từ file", type="primary", use_container_width=True):
                    with st.spinner(f"⏳ Đang phân tích {len(uploaded_files)} file bằng AI-OCR... (lần đầu có thể mất 1-2 phút để tải model)"):
                        try:
                            # Gộp kết quả OCR từ tất cả ảnh
                            merged_scores: dict[str, float] = {}
                            all_matched_lines: list[dict] = []
                            all_unmatched_lines: list[str] = []
                            all_raw_text: list[str] = []
                            all_warnings: list[str] = []
                            
                            for idx, uploaded_file in enumerate(uploaded_files):
                                uploaded_file.seek(0)  # Reset file pointer
                                
                                st.toast(f"🔄 Đang xử lý file {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")
                                
                                if uploaded_file.name.lower().endswith(".pdf"):
                                    # Xử lý PDF
                                    save_csv_path = os.path.join("data", "processed", f"{os.path.splitext(uploaded_file.name)[0]}.csv")
                                    
                                    import fitz
                                    try:
                                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                                        uploaded_file.seek(0)
                                        
                                        pdf_text = ""
                                        for page in doc:
                                            pdf_text += page.get_text()
                                            
                                        if len(pdf_text.strip()) > 20:
                                            # PDF chứa text
                                            ocr_result_single = extract_scores_from_student_pdf(uploaded_file, save_csv_path=save_csv_path)
                                        else:
                                            # PDF chứa ảnh (scan)
                                            ocr_result_single = {"scores": {}, "raw_text": [], "matched_lines": [], "unmatched_lines": [], "warnings": []}
                                            for i, page in enumerate(doc):
                                                pix = page.get_pixmap(dpi=150)
                                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                
                                                page_result = extract_scores_from_image(img)
                                                
                                                for subject, score in page_result["scores"].items():
                                                    if subject not in ocr_result_single["scores"] or score > ocr_result_single["scores"][subject]:
                                                        ocr_result_single["scores"][subject] = score
                                                        
                                                ocr_result_single["raw_text"].extend(page_result.get("raw_text", []))
                                                ocr_result_single["matched_lines"].extend(page_result.get("matched_lines", []))
                                                ocr_result_single["unmatched_lines"].extend(page_result.get("unmatched_lines", []))
                                                
                                            if not ocr_result_single["scores"]:
                                                ocr_result_single["warnings"].append("⚠️ Không trích xuất được điểm nào từ PDF ảnh.")
                                                
                                            if ocr_result_single["scores"] and save_csv_path:
                                                os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
                                                df = pd.DataFrame(list(ocr_result_single["scores"].items()), columns=["Tên môn", "Điểm"])
                                                df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
                                                
                                    except Exception as e:
                                        st.error(f"Lỗi đọc PDF: {e}")
                                        ocr_result_single = {"scores": {}, "raw_text": [], "matched_lines": [], "unmatched_lines": [], "warnings": [f"Lỗi: {e}"]}
                                else:
                                    # Xử lý Ảnh
                                    image = Image.open(uploaded_file)
                                    ocr_result_single = extract_scores_from_image(image)
                                
                                # Gộp điểm — giữ điểm cao nhất nếu trùng môn
                                for subject, score in ocr_result_single["scores"].items():
                                    if subject not in merged_scores or score > merged_scores[subject]:
                                        merged_scores[subject] = score
                                
                                # Gộp thông tin debug
                                for ml in ocr_result_single.get("matched_lines", []):
                                    ml["source_image"] = uploaded_file.name
                                    all_matched_lines.append(ml)
                                
                                for ul in ocr_result_single.get("unmatched_lines", []):
                                    all_unmatched_lines.append(f"[{uploaded_file.name}] {ul}")
                                
                                all_raw_text.extend(ocr_result_single.get("raw_text", []))
                            
                            # Tạo warnings tổng hợp
                            if not merged_scores:
                                all_warnings.append(
                                    "⚠️ Không trích xuất được điểm nào từ tất cả ảnh. "
                                    "Vui lòng kiểm tra chất lượng ảnh."
                                )
                            elif len(merged_scores) < 5:
                                all_warnings.append(
                                    f"⚠️ Chỉ trích xuất được {len(merged_scores)}/24 môn từ {len(uploaded_files)} ảnh. "
                                    "Kết quả có thể không chính xác."
                                )
                            
                            # Lưu kết quả gộp vào session state
                            merged_result = {
                                "scores": merged_scores,
                                "raw_text": all_raw_text,
                                "matched_lines": all_matched_lines,
                                "unmatched_lines": all_unmatched_lines,
                                "warnings": all_warnings,
                                "num_images": len(uploaded_files),
                            }
                            
                            st.session_state.ocr_result = merged_result
                            st.session_state.ocr_scores = merged_scores.copy()
                            st.session_state.step = 2.5
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi khi đọc ảnh: {e}")
            else:
                st.markdown(
                    "#### 💡 Mẹo để OCR chính xác hơn\n"
                    "- Ảnh nên rõ nét, không bị mờ hoặc nghiêng quá nhiều\n"
                    "- Đảm bảo tên môn học và điểm số đều hiển thị rõ ràng\n"
                    "- **Bảng điểm dài?** Chụp nhiều ảnh, rồi chọn tất cả cùng lúc khi upload\n"
                    "- Giữ **Ctrl** (Windows) hoặc **Cmd** (Mac) để chọn nhiều file\n"
                )
            
            st.divider()
            col_back2, col_manual = st.columns(2)
            with col_back2:
                if st.button("⬅️ Quay lại", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
            with col_manual:
                if st.button("⌨️ Nhập điểm thủ công (không cần ảnh)", use_container_width=True):
                    # Bỏ qua OCR, vào thẳng form nhập điểm với điểm trống
                    st.session_state.ocr_result = {}
                    st.session_state.ocr_scores = {}
                    st.session_state.step = 2.5
                    st.rerun()

        # --- STEP 2.5: REVIEW & CHỈNH SỬA ĐIỂM OCR ---
        elif st.session_state.step == 2.5:
            st.subheader("✏️ Kiểm tra & Chỉnh sửa điểm đã trích xuất")
            
            ocr_result = st.session_state.get('ocr_result', {})
            ocr_scores = st.session_state.get('ocr_scores', {})
            
            # Hiển thị cảnh báo nếu có
            for warning in ocr_result.get('warnings', []):
                st.warning(warning)
            
            # Thống kê OCR
            n_extracted = len(ocr_scores)
            n_total = len(STANDARD_SUBJECTS)
            n_images = ocr_result.get('num_images', 1)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("🖼️ Số ảnh đã xử lý", n_images)
            with col_stat2:
                st.metric("📊 Số môn trích xuất", f"{n_extracted}/{n_total}")
            with col_stat3:
                st.metric("✅ Số dòng match", len(ocr_result.get('matched_lines', [])))
            with col_stat4:
                st.metric("❓ Không nhận diện", len(ocr_result.get('unmatched_lines', [])))
            
            # Chi tiết OCR (expandable)
            with st.expander("🔍 Xem chi tiết OCR (Debug)"):
                if ocr_result.get('matched_lines'):
                    st.markdown("**Các dòng đã match thành công:**")
                    for ml in ocr_result['matched_lines']:
                        conf_emoji = "🟢" if ml['confidence'] > 0.7 else "🟡" if ml['confidence'] > 0.4 else "🔴"
                        source = f" 📎 _{ml['source_image']}_" if ml.get('source_image') else ""
                        st.markdown(
                            f"- {conf_emoji} **{ml['subject']}** = `{ml['score']}` "
                            f"(OCR: _{ml['ocr_text']}_){source}"
                        )
                
                if ocr_result.get('unmatched_lines'):
                    st.markdown("\n**Các dòng không nhận diện được:**")
                    for ul in ocr_result['unmatched_lines']:
                        st.markdown(f"- ❓ _{ul}_")
            
            st.divider()
            st.markdown(
                "#### 📝 Chỉnh sửa điểm\n"
                "Điểm đã được trích xuất tự động từ ảnh. "
                "Bạn có thể chỉnh sửa nếu OCR đọc sai, hoặc bỏ chọn nếu chưa học môn đó."
            )
            
            # Hiển thị form chỉnh sửa điểm
            scores_dict = {}
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    has_score = feature in ocr_scores
                    default_score = ocr_scores.get(feature, 0.0)
                    
                    # Hiển thị icon cho môn đã trích xuất vs chưa
                    icon = "✅" if has_score else "⬜"
                    st.markdown(f"{icon} **{feature}**")
                    
                    # Checkbox: đã có điểm hay chưa
                    studied_key = f"ocr_studied_{feature}"
                    is_studied = st.checkbox(
                        "Đã có điểm", 
                        value=has_score, 
                        key=f"ocr_check_{feature}"
                    )
                    
                    input_val = st.number_input(
                        f"Điểm {feature}", 
                        min_value=MIN_SCORE, 
                        max_value=MAX_SCORE,
                        value=float(default_score) if has_score else 0.0, 
                        step=0.1, 
                        key=f"ocr_input_{feature}",
                        label_visibility="collapsed", 
                        disabled=not is_studied
                    )
                    
                    if is_studied:
                        scores_dict[feature] = input_val
                    else:
                        scores_dict[feature] = -1.0
                    
                    st.markdown("---")
            
            st.divider()
            col_back, col_reupload, col_next = st.columns([1, 1, 3])
            with col_back:
                if st.button("⬅️ Quay lại", use_container_width=True, key="back_from_review"):
                    st.session_state.step = 1
                    st.rerun()
            with col_reupload:
                if st.button("📸 Upload ảnh khác", use_container_width=True):
                    # Reset OCR state
                    if 'ocr_result' in st.session_state:
                        del st.session_state.ocr_result
                    if 'ocr_scores' in st.session_state:
                        del st.session_state.ocr_scores
                    st.session_state.step = 2
                    st.rerun()
            with col_next:
                if st.button("🔮 Hoàn tất & Xem Khuyến nghị", type="primary", use_container_width=True):
                    is_valid, err_msg = validate_scores(scores_dict)
                    if not is_valid:
                        st.error(err_msg)
                    else:
                        st.session_state.scores_dict = scores_dict
                        st.session_state.step = 3
                        st.rerun()

        # --- STEP 3: RESULTS & FEEDBACK ---
        elif st.session_state.step == 3:
            user_name = st.session_state.user_name
            selected_prefs = st.session_state.selected_prefs
            scores_dict = st.session_state.scores_dict
            
            if st.button("⬅️ Chỉnh sửa thông tin/điểm", key="back_to_2"):
                st.session_state.step = 2.5 if st.session_state.get('ocr_result') else 2
                st.rerun()
                
            input_df = pd.DataFrame([scores_dict])
            
            # --- AI Prediction ---
            classes = model.classes_
            probabilities_raw = model.predict_proba(input_df)[0]
            
            # --- TÍNH ĐIỂM HYBRID SCORE ---
            hybrid_scores = dict(zip(classes, probabilities_raw))
            
            for pref_career in selected_prefs:
                if pref_career in hybrid_scores:
                    hybrid_scores[pref_career] += PREFERENCE_BONUS
            
            total_score = sum(hybrid_scores.values())
            for k in hybrid_scores:
                hybrid_scores[k] = (hybrid_scores[k] / total_score) * 100
                
            # --- RANKING ---
            ranked_careers = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            top1_career = ranked_careers[0][0]
            top1_score = ranked_careers[0][1]
            
            st.header(f"🎯 Kết quả Tư vấn cho {user_name if user_name else 'bạn'}")
            
            st.success(f"### 🏆 Chuyên ngành phù hợp nhất: **{top1_career}**\n**Độ phù hợp (Năng lực + Sở thích):** {top1_score:.1f}%")
            
            st.markdown("#### 🥇 Bảng xếp hạng chi tiết")
            rank_col1, rank_col2, rank_col3 = st.columns(3)
            with rank_col1:
                st.metric(label="Top 1", value=ranked_careers[0][0], delta=f"{ranked_careers[0][1]:.1f}%")
            with rank_col2:
                st.metric(label="Top 2", value=ranked_careers[1][0], delta=f"{ranked_careers[1][1]:.1f}%", delta_color="off")
            with rank_col3:
                st.metric(label="Top 3", value=ranked_careers[2][0], delta=f"{ranked_careers[2][1]:.1f}%", delta_color="off")
            
            st.markdown("#### 📝 Lời khuyên hành động")
            st.info(CAREER_ADVICE.get(top1_career, "Vui lòng tập trung học tốt các môn chuyên ngành."))
            
            # Tính explanation trước expander để tránh NameError khi tải Excel
            explanation_steps = explain_decision(model, input_df)
            
            with st.expander("🧠 Xem chi tiết quá trình phân tích AI (XAI)"):
                st.caption("AI đã lập luận thế nào dựa trên điểm số bạn nhập:")
                if explanation_steps:
                    list_md = ""
                    for i, step in enumerate(explanation_steps):
                        emoji = "✅" if step['operator'] == ">" else "🔻"
                        if step['score'] == -1.0:
                            list_md += f"- {emoji} Node {i+1}: Môn **{step['subject']}** ở trạng thái `Chưa học` (Thỏa mãn `{step['operator']} {step['threshold']:.2f}`)\n"
                        else:
                            list_md += f"- {emoji} Node {i+1}: Điểm môn **{step['subject']}** là `{step['score']:.1f}` (Thỏa mãn `{step['operator']} {step['threshold']:.2f}`)\n"
                    st.markdown(list_md)
                else:
                    st.warning("Không thể phân tích đường ra quyết định.")
                
                st.plotly_chart(create_bar_chart(scores_dict), use_container_width=True)
            
            excel_bytes = generate_excel_report(user_name, scores_dict, top1_career, explanation_steps)
            st.download_button("📥 Tải xuống Báo cáo Excel", data=excel_bytes, file_name="DSS_Report.xlsx")
            
            st.divider()
            st.subheader("⭐ Đánh giá trải nghiệm của bạn")
            st.markdown("Hệ thống có đưa ra gợi ý hữu ích cho bạn không?")
            
            with st.form("feedback_form"):
                rating = st.radio("Đánh giá sao (1 = Rất tệ, 5 = Rất hữu ích):", [1, 2, 3, 4, 5], index=4, horizontal=True)
                comment = st.text_input("Góp ý thêm (Tùy chọn):")
                submit_fb = st.form_submit_button("Gửi đánh giá")
                
                if submit_fb:
                    save_feedback(user_name, top1_career, rating, comment)
                    st.success("Cảm ơn bạn đã đánh giá! Vui lòng tải lại trang để thấy điểm đánh giá được cập nhật.")

    with tab2:
        st.subheader("📊 Phân Tích Mô Hình Học Máy")
        st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix)")
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
            
        st.markdown("#### Cấu trúc Cây quyết định (Tree Plot)")
        if os.path.exists(TREE_PLOT_PATH):
            st.image(TREE_PLOT_PATH, use_container_width=True)


if __name__ == "__main__":
    main()
