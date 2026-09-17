"""
app.py — Streamlit Dashboard (Main Entry Point)
============================================================
Giao diện Web cho Hệ Trợ Giúp Quyết Định Học Tập FIT-HAU.

Features:
    - Tab 1: Trợ giúp quyết định (Nhập điểm → Dự đoán + XAI)
    - Tab 2: Phân tích (Confusion Matrix + Tree Plot)
    - Sidebar: Thông tin & cấu hình
    - Export: Download phiếu kết quả Excel

Architecture (Hybrid 2 tầng):
    - Tầng 1: Content-Based Filtering (Cosine Similarity) → Gán nhãn
    - Tầng 2: Decision Tree → Dự đoán + XAI
    - KHÔNG có logic training — chỉ load model .pkl và gọi predict()
    - Form nhập liệu ĐỘNG theo feature_names_in_ của mô hình
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# ============================================================
# CONSTANTS
# ============================================================
MODEL_PATH: str = os.path.join("models", "dss_brain.pkl")
CONFUSION_MATRIX_PATH: str = os.path.join("reports", "dss_confusion_matrix.png")
TREE_PLOT_PATH: str = os.path.join("reports", "dss_tree.png")

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0


# ============================================================
# MODEL LOADING (cached)
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


# ============================================================
# VALIDATION
# ============================================================
def validate_scores(scores: dict[str, float]) -> bool:
    """Zero-Trust validation: block invalid scores immediately."""
    for subject, score in scores.items():
        if score < MIN_SCORE or score > MAX_SCORE:
            st.error(f"🚨 Điểm số không hợp lệ ở môn {subject}: {score}. Vui lòng nhập điểm từ {MIN_SCORE} đến {MAX_SCORE}.")
            st.stop()
    return True


# ============================================================
# EXPLAINABLE AI (XAI)
# ============================================================
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


# ============================================================
# VISUALIZATION
# ============================================================
def create_bar_chart(scores: dict[str, float]) -> go.Figure:
    """Create a Horizontal Bar Chart visualizing student scores."""
    # Sắp xếp điểm giảm dần
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
        height=max(400, len(subjects) * 30), # Chiều cao linh hoạt theo số môn
        margin=dict(l=200, r=20, t=40, b=20)
    )
    return fig


# ============================================================
# EXPORT
# ============================================================
def generate_excel_report(student_data: dict, prediction: str, explanation_steps: list[dict]) -> bytes:
    """Generate an Excel report in-memory using io.BytesIO()."""
    df = pd.DataFrame([student_data])
    df["Dự đoán chuyên ngành"] = prediction
    
    # Format list of dicts to string for excel
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
# MAIN APP
# ============================================================
def main() -> None:
    # --- Page Config ---
    st.set_page_config(
        page_title="DSS FIT-HAU — Trợ Giúp Quyết Định Học Tập",
        page_icon="🎓",
        layout="wide",
    )

    # Nạp mô hình AI
    model = load_model()
    features = model.feature_names_in_

    # --- Sidebar ---
    with st.sidebar:
        st.title("🎓 DSS FIT-HAU")
        st.markdown("**Hệ Trợ Giúp Quyết Định Học Tập**")
        st.markdown("Khoa CNTT — ĐH Kiến trúc Hà Nội")
        st.divider()
        st.caption("Powered by Content-Based Filtering + Decision Tree + XAI")

    # --- Main Content ---
    st.title("🎓 Hệ Trợ Giúp Quyết Định Học Tập")
    st.markdown(f"Nhập điểm {len(features)} môn nền tảng để nhận gợi ý chuyên ngành phù hợp.")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🧭 Trợ Giúp Quyết Định", "📊 Phân Tích Mô Hình"])

    with tab1:
        st.subheader("Nhập điểm số")
        st.info("💡 Nhập điểm từ 0.0 đến 10.0 cho từng môn học.")

        # Render form động theo danh sách features từ mô hình
        scores_dict = {}
        cols = st.columns(3) # Hiển thị thành 3 cột
        for i, feature in enumerate(features):
            with cols[i % 3]:
                scores_dict[feature] = st.number_input(
                    feature, 
                    min_value=MIN_SCORE, 
                    max_value=MAX_SCORE, 
                    value=0.0, 
                    step=0.1,
                    key=f"input_{feature}"
                )
            
        if st.button("🔍 Phân tích & Gợi ý", type="primary"):
            # Zero-trust validation
            validate_scores(scores_dict)
            
            input_df = pd.DataFrame([scores_dict])
            
            # Dự đoán
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_
            explanation_steps = explain_decision(model, input_df)
            
            st.divider()
            st.subheader("🎯 Kết quả Tư vấn")
            
            # Khối kết quả chính
            max_prob = max(probabilities) * 100
            st.success(f"### **Chuyên ngành phù hợp nhất:** {prediction}\n**Độ tin cậy của AI:** {max_prob:.1f}%")
            
            # Phân bố xác suất
            with st.expander("📊 Phân bố xác suất cho tất cả các ngành", expanded=False):
                for cls, prob in zip(classes, probabilities):
                    col1, col2 = st.columns([1, 4])
                    col1.write(f"**{cls}**")
                    col2.progress(float(prob), text=f"{prob*100:.1f}%")
            
            st.markdown("### 🧠 Quá trình lập luận của AI (XAI)")
            st.caption("Cách trí tuệ nhân tạo suy luận để đưa ra quyết định dựa trên bộ não Decision Tree:")
            
            if explanation_steps:
                list_md = ""
                for i, step in enumerate(explanation_steps):
                    emoji = "✅" if step['operator'] == ">" else "🔻"
                    list_md += f"- {emoji} **Bước {i+1}:** Nhận thấy điểm môn **{step['subject']}** là `{step['score']:.1f}` (Thỏa mãn điều kiện rẽ nhánh `{step['operator']} {step['threshold']:.2f}`)\n"
                st.info(list_md)
            else:
                st.warning("Không thể phân tích đường ra quyết định.")
            
            # Biểu đồ điểm số
            bar_fig = create_bar_chart(scores_dict)
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Xuất Excel
            excel_bytes = generate_excel_report(scores_dict, prediction, explanation_steps)
            st.download_button(
                label="📥 Tải xuống Báo cáo Excel",
                data=excel_bytes,
                file_name="DSS_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tab2:
        st.subheader("📊 Phân Tích Mô Hình AI")
        
        st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix)")
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, use_container_width=True)
        else:
            st.warning("Không tìm thấy ảnh Confusion Matrix.")
            
        st.markdown("#### Cấu trúc Cây quyết định (Tree Plot)")
        if os.path.exists(TREE_PLOT_PATH):
            st.image(TREE_PLOT_PATH, use_container_width=True)
        else:
            st.warning("Không tìm thấy ảnh Tree Plot.")


if __name__ == "__main__":
    main()
