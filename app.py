import streamlit as st
import os
import io
from PIL import Image
from src.detector import FloorPlanDetector
from src.visualizer import Visualizer
from src.compliance import ComplianceAnalyzer

st.set_page_config(layout="wide", page_title="Floor Plan POC", page_icon="🏢")

# --- Initialize Engines ---
# We use st.cache_resource so the YOLO model doesn't reload on every button click
@st.cache_resource
def load_detector():
    return FloorPlanDetector()

@st.cache_resource
def load_analyzer():
    return ComplianceAnalyzer()

try:
    detector = load_detector()
    analyzer = load_analyzer()
    models_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {e}")
    models_loaded = False

# --- UI Layout ---
st.title("🏢 Generic Floor Plan Analysis POC")
st.markdown("**Stage 1:** YOLOv8 Structural Detection ➡️ **Stage 2:** GPT-4o Semantic Compliance")

uploaded_file = st.file_uploader("Upload a floor plan image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and models_loaded:
    # Save the uploaded file temporarily so YOLO can read it
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Upload")
        st.image(temp_path, use_container_width=True)

    if st.button("🚀 Run Analysis Pipeline", type="primary", use_container_width=True):
        st.divider()
        
        # --- STAGE 1: YOLOv8 ---
        with st.spinner("Stage 1: YOLOv8 detecting structures..."):
            raw_result = detector.detect(temp_path)
            elements = detector.extract_elements(raw_result)
            
            # Draw boxes using our Visualizer
            annotated_img = Visualizer.draw_bounding_boxes(temp_path, elements)

        with col2:
            st.markdown("### Stage 1: YOLOv8 Detections")
            st.image(annotated_img, use_container_width=True)
            st.success(f"Detected {len(elements)} structural elements.")

        # --- STAGE 2: GPT-4o ---
        if elements:
            with st.spinner("Stage 2: GPT-4o analyzing compliance..."):
                report = analyzer.analyze(elements)

            st.markdown("### Stage 2: Compliance Report")
            
            if "error" in report:
                st.error(f"GPT-4o Error: {report['error']}")
            else:
                matrix = report.get('explainability_matrix', [])
                
                # ── THE DETERMINISTIC OVERRIDE ──
                # 1. We ignore the LLM's hallucinated score.
                # 2. We calculate the true score by summing the point impacts in the matrix.
                total_penalties = sum(item.get('point_impact', 0) for item in matrix)
                true_score = max(0, 100 + total_penalties) # Ensures score doesn't drop below 0
                
                color = "green" if true_score >= 80 else "orange" if true_score >= 50 else "red"
                
                st.markdown(f"**Final Score:** :{color}[{true_score}/100] &nbsp;&nbsp;|&nbsp;&nbsp; **Status:** {report.get('status', 'Unknown')}")
                st.markdown(f"**Summary:** {report.get('summary', '')}")
                
                # --- SHAP-LIKE EXPLAINABILITY UI ---
                if matrix:
                    st.divider()
                    st.markdown("#### 🔍 Score Explainability (Penalty Matrix)")
                    st.caption("Starting Base Score: 100")
                    
                    for item in matrix:
                        col_text, col_score = st.columns([4, 1])
                        with col_text:
                            st.markdown(f"**{item.get('feature', 'Unknown')}** — {item.get('issue', '')}")
                            st.caption(item.get('reasoning', ''))
                        with col_score:
                            st.metric(label="", value="", delta=item.get('point_impact', 0))
                        st.write("---")
                else:
                    st.success("No compliance issues detected. Perfect score!")
                    
                with st.expander("View Raw JSON & Extracted Elements"):
                    st.json(report)
                    st.write("YOLO Output sent to GPT-4o:")
                    st.json(elements)
        else:
            st.warning("No elements detected to analyze. Check YOLO model confidence.")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)