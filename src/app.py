"""
app.py — Sprint 3: Streamlit Dashboard (Main Entry Point)
============================================================
Giao diện Web cho Hệ Trợ Giúp Quyết Định Học Tập FIT-HAU.

Features:
    - Tab 1: Trợ giúp quyết định (Nhập điểm → Dự đoán + XAI)
    - Tab 2: Phân tích (Confusion Matrix + Tree Plot)
    - Sidebar: Thông tin & cấu hình
    - Export: Download phiếu kết quả Excel

Architecture:
    - KHÔNG có logic training — chỉ load model .pkl và gọi predict()
    - Zero-Trust Validation trên mọi input
    - Explainable AI qua decision_path
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

FEATURE_COLUMNS: list[str] = ["Toan_Roi_Rac", "Lap_Trinh_C", "Co_So_Du_Lieu"]
FEATURE_LABELS: list[str] = ["Toán Rời Rạc", "Lập Trình C", "Cơ Sở Dữ Liệu"]

MAX_SCORE: float = 10.0
MIN_SCORE: float = 0.0


# ============================================================
# MODEL LOADING (cached — load once into RAM)
# ============================================================
@st.cache_resource
def load_model():
    """Load trained Decision Tree model from .pkl file.

    Uses @st.cache_resource to avoid reloading on every rerun.
    """
    # TODO: Sprint 3 - Task 1
    raise NotImplementedError("Sprint 3: Implement model loading")


# ============================================================
# VALIDATION
# ============================================================
def validate_scores(scores: dict[str, float]) -> bool:
    """Zero-Trust validation: block invalid scores immediately.

    Rules:
        - All scores must be >= 0.0 and <= 10.0
        - Calls st.stop() on violation
    """
    # TODO: Sprint 3 - Task 2
    raise NotImplementedError("Sprint 3 - Task 2: Implement input validation")


# ============================================================
# EXPLAINABLE AI (XAI)
# ============================================================
def explain_decision(model, input_data: pd.DataFrame) -> str:
    """Trace the decision_path of the Decision Tree and translate to Vietnamese.

    Returns:
        str: Human-readable explanation of the AI's reasoning.
    """
    # TODO: Sprint 3 - Task 3
    raise NotImplementedError("Sprint 3 - Task 3: Implement XAI decision path")


# ============================================================
# VISUALIZATION
# ============================================================
def create_radar_chart(scores: dict[str, float]) -> go.Figure:
    """Create a Radar Chart visualizing student skill profile.

    Uses Plotly for interactive chart.
    """
    # TODO: Sprint 3 - Task 1 (part of UI)
    raise NotImplementedError("Sprint 3: Implement radar chart")


# ============================================================
# EXPORT
# ============================================================
def generate_excel_report(student_data: dict, prediction: str, explanation: str) -> bytes:
    """Generate an Excel report in-memory using io.BytesIO().

    MUST NOT save physical file to disk — use RAM buffer only.

    Returns:
        bytes: Excel file content for download.
    """
    # TODO: Sprint 3 - Task from Epic 4
    raise NotImplementedError("Sprint 3: Implement Excel export")


# ============================================================
# MAIN APP
# ============================================================
def main() -> None:
    """Main Streamlit application entry point."""

    # --- Page Config ---
    st.set_page_config(
        page_title="DSS FIT-HAU — Trợ Giúp Quyết Định Học Tập",
        page_icon="🎓",
        layout="wide",
    )

    # --- Sidebar ---
    with st.sidebar:
        st.title("🎓 DSS FIT-HAU")
        st.markdown("**Hệ Trợ Giúp Quyết Định Học Tập**")
        st.markdown("Khoa CNTT — ĐH Kiến trúc Hà Nội")
        st.divider()
        st.caption("Powered by Decision Tree + Explainable AI")

    # --- Main Content ---
    st.title("🎓 Hệ Trợ Giúp Quyết Định Học Tập")
    st.markdown("Nhập điểm 3 môn nền tảng để nhận gợi ý chuyên ngành phù hợp.")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🧭 Trợ Giúp Quyết Định", "📊 Phân Tích Mô Hình"])

    with tab1:
        st.subheader("Nhập điểm số")
        st.info("💡 Nhập điểm từ 0.0 đến 10.0 cho từng môn học.")

        # TODO: Sprint 3 — Implement score input form, prediction, XAI display,
        #       radar chart, and Excel download button here.
        st.warning("⚠️ Module đang trong quá trình phát triển (Sprint 3)")

    with tab2:
        st.subheader("📊 Phân Tích Mô Hình AI")

        # TODO: Sprint 3 - Task 4 — Display confusion matrix and tree plot images
        st.warning("⚠️ Module đang trong quá trình phát triển (Sprint 3)")


if __name__ == "__main__":
    main()
