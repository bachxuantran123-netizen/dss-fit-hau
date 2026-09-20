"""
app.py — Streamlit Dashboard (Main Entry Point)
============================================================
Giao diện Web cho Hệ Trợ Giúp Quyết Định Học Tập FIT-HAU.

Architecture (Hybrid 2 tầng + Rule-based Preferences):
    - Tầng 1: Content-Based Filtering (Cosine Similarity) → Gán nhãn
    - Tầng 2: Decision Tree → Dự đoán + XAI
    - Tầng 3 (Mới): Hybrid Score = AI Score + Preference Bonus
"""

import os
import io
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# ============================================================
# CONSTANTS & CONFIGS
# ============================================================
MODEL_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")
FEEDBACK_PATH: str = os.path.join("data", "feedback.csv")

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0

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
    """Zero-Trust validation: kiểm tra điểm hợp lệ."""
    for subject, score in scores.items():
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
    
    # Xử lý tên để tránh gãy CSV nếu có dấu phẩy
    safe_name = name.replace(',', '') if name else "Ẩn danh"
    safe_comment = comment.replace(',', '') if comment else "Không có"
    
    with open(FEEDBACK_PATH, "a", encoding='utf-8-sig') as f:
        if not file_exists:
            f.write("Timestamp,Name,Suggested_Career,Rating,Comment\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{safe_name},{suggested_career},{rating},{safe_comment}\n")


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

        # --- STEP 2: SCORE INPUT ---
        elif st.session_state.step == 2:
            st.subheader("📚 Năng lực học tập")
            greeting_name = st.session_state.user_name if st.session_state.user_name else "bạn"
            st.info(f"Chào **{greeting_name}**, vui lòng nhập điểm cho các môn đã học. Tích bỏ chọn nếu chưa học môn đó.")

            scores_dict = {}
            cols = st.columns(3)
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    st.markdown(f"**{feature}**")
                    
                    # Checkbox to toggle studied state
                    studied_key = f"studied_{feature}"
                    is_studied = st.checkbox("Đã có điểm", value=st.session_state.get(studied_key, True), key=f"check_{feature}")
                    
                    if is_studied:
                        score_val = st.session_state.get(f"score_{feature}", 0.0)
                        scores_dict[feature] = st.number_input(
                            f"Điểm {feature}", min_value=MIN_SCORE, max_value=MAX_SCORE, 
                            value=score_val, step=0.1, key=f"input_score_{feature}",
                            label_visibility="collapsed"
                        )
                    else:
                        st.caption("*(Bỏ qua / Chưa học)*")
                        scores_dict[feature] = -1.0
                        
                    st.markdown("---")
            
            st.divider()
            col_back, col_next = st.columns([1, 4])
            with col_back:
                if st.button("⬅️ Quay lại", use_container_width=True):
                    for feature in features:
                        is_stud = st.session_state[f"check_{feature}"]
                        st.session_state[f"studied_{feature}"] = is_stud
                        if is_stud:
                            st.session_state[f"score_{feature}"] = st.session_state[f"input_score_{feature}"]
                        else:
                            st.session_state[f"score_{feature}"] = -1.0
                    st.session_state.step = 1
                    st.rerun()
            with col_next:
                if st.button("🔮 Hoàn tất & Xem Khuyến nghị", type="primary", use_container_width=True):
                    for feature in features:
                        is_stud = st.session_state[f"check_{feature}"]
                        st.session_state[f"studied_{feature}"] = is_stud
                        if is_stud:
                            st.session_state[f"score_{feature}"] = st.session_state[f"input_score_{feature}"]
                        else:
                            st.session_state[f"score_{feature}"] = -1.0
                    
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
                st.session_state.step = 2
                st.rerun()
                
            input_df = pd.DataFrame([scores_dict])
            
            # --- AI Prediction ---
            classes = model.classes_
            probabilities_raw = model.predict_proba(input_df)[0]
            
            # --- TÍNH ĐIỂM HYBRID SCORE ---
            hybrid_scores = dict(zip(classes, probabilities_raw))
            BONUS_VALUE = 0.10 
            
            for pref_career in selected_prefs:
                if pref_career in hybrid_scores:
                    hybrid_scores[pref_career] += BONUS_VALUE
            
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
            
            with st.expander("🧠 Xem chi tiết quá trình phân tích AI (XAI)"):
                st.caption("AI đã lập luận thế nào dựa trên điểm số bạn nhập:")
                explanation_steps = explain_decision(model, input_df)
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
