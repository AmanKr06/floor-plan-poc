import streamlit as st
import os
import io
from PIL import Image
from src.detector import FloorPlanDetector
from src.visualizer import Visualizer
from src.compliance import ComplianceAnalyzer
from src.shap_explainer import (
    extract_features,
    run_shap,
    render_waterfall_chart,
    render_feature_table,
    SHAP_AVAILABLE,
)
import pandas as pd

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

def render_shap_section(detections: list, compliance_result: dict, true_score: int):
    """
    Stage 3: SHAP explainability section.
    Call this after you've displayed the Stage 2 (GPT-4o) compliance results.

    Args:
        detections       — output of detector.extract_elements()
        compliance_result — output of compliance.analyze()
    """
    st.divider()
    # Change the header to explicitly call out XGBoost
    st.markdown("### Stage 3 & 4 — XGBoost Prediction & SHAP Explainability")
    st.markdown(
        "**Stage 3:** XGBoost Meta-Model calculates the final feasibility score. \n"
        "**Stage 4:** SHAP explains *why* XGBoost gave that score."
    )

    if not SHAP_AVAILABLE:
        st.error(
            "❌ SHAP not installed. Run:  `pip install shap matplotlib`  "
            "then restart Streamlit."
        )
        return

    if not detections:
        st.warning("No detections to explain. Run YOLO detection first.")
        return

    # ── Extract numeric features from the YOLO detections ────────────────────
    with st.spinner("Computing XGBoost + SHAP values..."):
        # Pass the report and score to extract_features
        features = extract_features(detections, compliance_result, true_score)
        shap_result = run_shap(features)

    if "error" in shap_result:
        st.error(f"SHAP error: {shap_result['error']}")
        return

    # ── Layout: score card + feature table on the left, chart on the right ───
    col_left, col_right = st.columns([1, 2])

    with col_left:
        score     = shap_result["predicted"]
        base      = shap_result["base_value"]
        score_clr = "#166534" if score >= 70 else ("#854D0E" if score >= 45 else "#991B1B")

        st.markdown(f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;
                    padding:16px 20px;margin-bottom:12px">
            <div style="font-size:11px;color:#6B7280;margin-bottom:4px">
                XGBoost Final Feasibility Score
            </div>
            <div style="font-size:44px;font-weight:800;color:{score_clr};line-height:1">
                {score:.0f}
                <span style="font-size:18px;color:#9CA3AF">/100</span>
            </div>
            <div style="font-size:11px;color:#9CA3AF;margin-top:6px">
                Base (avg floor plan): {base:.1f}
            </div>
            <div style="font-size:11px;color:#9CA3AF">
                Score shift: {score - base:+.1f} points
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Cross-reference with the True GPT-4o score
        if true_score is not None:
            delta = score - true_score
            st.markdown(
                f"**GPT-4o Visual Score:** {true_score}/100  \n"
                f"**XGBoost Meta Score:** {score:.0f}/100  \n"
                f"**Difference:** {delta:+.1f} pts"
            )
            if abs(delta) < 10:
                st.success("✅ Models broadly agree — high confidence in analysis")
            else:
                st.warning(
                    "⚠️ Scores diverge by more than 10 pts. "
                    "XGBoost is weighing the strict geometric rules against GPT-4o's semantic reasoning."
                )

        # Feature value table
        st.markdown("**Feature breakdown**")
        table_rows = render_feature_table(features, shap_result["shap_values"])
        df = pd.DataFrame(table_rows)

        # Colour the Direction column
        def colour_direction(val):
            if "↑" in val:  return "color: #166534; font-weight: 600"
            if "↓" in val:  return "color: #991B1B; font-weight: 600"
            return "color: #6B7280"

        st.dataframe(
            df.style.map(colour_direction, subset=["Direction"]),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        # Waterfall chart
        st.markdown("**Waterfall chart** — feature contributions to the score")
        chart_bytes = render_waterfall_chart(shap_result)
        st.image(chart_bytes, use_container_width=True)

        # Narrative explanation
        sv = shap_result["shap_values"]
        feature_names = shap_result["feature_names"]
        top_neg = [(feature_names[i], sv[i]) for i in range(len(sv)) if sv[i] < -1]
        top_pos = [(feature_names[i], sv[i]) for i in range(len(sv)) if sv[i] > 1]

        if top_neg or top_pos:
            narrative_parts = []
            if top_neg:
                worst = min(top_neg, key=lambda x: x[1])
                narrative_parts.append(
                    f"The biggest score penalty came from **{worst[0].replace('_', ' ')}** "
                    f"({worst[1]:.1f} pts)."
                )
            if top_pos:
                best = max(top_pos, key=lambda x: x[1])
                narrative_parts.append(
                    f"The biggest positive contributor was **{best[0].replace('_', ' ')}** "
                    f"(+{best[1]:.1f} pts)."
                )
            st.info("  ".join(narrative_parts))    

# --- UI Layout ---
st.title("🏢 Generic Floor Plan Analysis POC")
st.markdown("**Stage 1:** YOLOv8 Structural Detection ➡️ **Stage 2:** GPT-4o Semantic Compliance ➡️ **Stage 3:** XGBoost ➡️ **Stage 4:** SHAP Explainability")

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
            
            # ── NEW: Save the annotated image so GPT-4o can read it ──
            annotated_path = f"annotated_{uploaded_file.name}"
            annotated_img.save(annotated_path)

        with col2:
            st.markdown("### Stage 1: YOLOv8 Detections")
            st.image(annotated_img, use_container_width=True)
            st.success(f"Detected {len(elements)} structural elements.")

        # --- STAGE 2: GPT-4o ---
        if elements:
            with st.spinner("Stage 2: GPT-4o Vision analyzing compliance..."):
                # ── NEW: Pass the annotated image to the analyzer ──
                report = analyzer.analyze(elements, annotated_path)

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

        # ── Stage 3 & 4: XGBoost & SHAP ────────────────────────
        if elements and 'true_score' in locals():
            render_shap_section(elements, report, true_score)

    # Cleanup temp files
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # NEW: Also clean up the annotated image we created for Vision!
    if 'annotated_path' in locals() and os.path.exists(annotated_path):
        os.remove(annotated_path)