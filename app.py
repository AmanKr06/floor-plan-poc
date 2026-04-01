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
                score = report.get('compliance_score', 0)
                color = "green" if score >= 80 else "orange" if score >= 50 else "red"
                
                st.markdown(f"**Score:** :{color}[{score}/100] &nbsp;&nbsp;|&nbsp;&nbsp; **Status:** {report.get('status', 'Unknown')}")
                st.markdown(f"**Summary:** {report.get('summary', '')}")
                
                issues = report.get('issues', [])
                if issues:
                    st.warning("**Identified Issues:**\n" + "\n".join([f"- {issue}" for issue in issues]))
                else:
                    st.success("No compliance issues detected based on available elements.")
                    
                with st.expander("View Raw JSON & Extracted Elements"):
                    st.json(report)
                    st.write("YOLO Output sent to GPT-4o:")
                    st.json(elements)
        else:
            st.warning("No elements detected to analyze. Check YOLO model confidence.")

    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)