import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
import xgboost as xgb
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── 1. UPDATED FEATURE SET (7 Features) ──
FEATURE_NAMES = [
    "zone_count",           
    "door_count",           
    "window_count",         
    "zones_missing_door",   
    "zones_missing_window", 
    "gpt_compliance_score", # NEW: Stage 2 input
    "gpt_status_encoded"    # NEW: Stage 2 input
]

FEATURE_LABELS = {
    "zone_count":           "Total Zones",
    "door_count":           "Total Doors",
    "window_count":         "Total Windows",
    "zones_missing_door":   "Zones Missing Door",
    "zones_missing_window": "Zones Missing Window",
    "gpt_compliance_score": "GPT-4o Vision Score",
    "gpt_status_encoded":   "GPT-4o Status (0=Fail, 2=Pass)"
}

# The baseline needs to match the 7 features.
# We assume a "perfect" baseline where GPT gave it a 100 (Pass).
BACKGROUND_MEANS = np.array([[
    3.0,  # zone_count
    3.0,  # door_count
    3.0,  # window_count
    0.0,  # zones_missing_door
    0.0,  # zones_missing_window
    100.0,# gpt_compliance_score
    2.0   # gpt_status_encoded (2 = Pass)
]])

# ── 2. LOAD THE XGBOOST BRAIN ──
try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("models/feasibility_xgb.ubj")
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading XGBoost model: {e}")

# ── Proximity helper ──
def _is_near_zone(element: dict, zone: dict, margin: int = 60) -> bool:
    ex = element["coordinates"]["x"]
    ey = element["coordinates"]["y"]
    zx = zone["coordinates"]["x"]
    zy = zone["coordinates"]["y"]
    zw = zone["coordinates"]["w"]
    zh = zone["coordinates"]["h"]
    return (zx - zw/2 - margin) <= ex <= (zx + zw/2 + margin) and \
           (zy - zh/2 - margin) <= ey <= (zy + zh/2 + margin)

# ── Feature extraction ──
# We now require the GPT report to build the full feature set.
def extract_features(detections: list, gpt_report: dict, gpt_true_score: int) -> dict:
    zones   = [d for d in detections if d["class_name"] == "zone"]
    doors   = [d for d in detections if d["class_name"] == "door"]
    windows = [d for d in detections if d["class_name"] == "window"]

    zones_missing_door = sum(1 for z in zones if not any(_is_near_zone(d, z) for d in doors))
    zones_missing_window = sum(1 for z in zones if not any(_is_near_zone(w, z) for w in windows))

    # Encode the status
    status_str = gpt_report.get("status", "Fail").lower()
    if status_str == "pass": status_encoded = 2.0
    elif status_str == "review": status_encoded = 1.0
    else: status_encoded = 0.0

    return {
        "zone_count":           float(len(zones)),
        "door_count":           float(len(doors)),
        "window_count":         float(len(windows)),
        "zones_missing_door":   float(zones_missing_door),
        "zones_missing_window": float(zones_missing_window),
        "gpt_compliance_score": float(gpt_true_score),
        "gpt_status_encoded":   float(status_encoded)
    }

# ── SHAP computation (NOW USING XGBOOST) ──
def run_shap(features_dict: dict) -> dict:
    if not SHAP_AVAILABLE: return {"error": "shap not installed."}
    if not MODEL_LOADED: return {"error": "XGBoost model not found at models/feasibility_xgb.ubj"}

    # Convert to 2D numpy array for the model
    feature_vector = np.array([[features_dict[f] for f in FEATURE_NAMES]])

    # Generate the prediction from our trained model
    predicted = xgb_model.predict(feature_vector)[0]

    # Use TreeExplainer (optimized specifically for XGBoost)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(feature_vector)
    
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    return {
        "shap_values":   sv,
        "base_value":    float(explainer.expected_value),
        "predicted":     float(predicted),
        "features":      features_dict,
        "feature_names": FEATURE_NAMES,
    }

# ── Chart rendering (Unchanged, just copied over) ──
def render_waterfall_chart(shap_result: dict) -> bytes:
    sv            = shap_result["shap_values"]
    base_value    = shap_result["base_value"]
    predicted     = shap_result["predicted"]
    features      = shap_result["features"]
    feature_names = shap_result["feature_names"]

    sorted_idx  = np.argsort(np.abs(sv))[::-1]
    sorted_sv   = sv[sorted_idx]
    sorted_names = [
        f"{FEATURE_LABELS.get(feature_names[i], feature_names[i])}"
        f"\n= {features[feature_names[i]]:.1f}"
        for i in sorted_idx
    ]

    n = len(sorted_sv)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.65 + 2)))
    fig.patch.set_facecolor("#F9FAFB")
    ax.set_facecolor("#F9FAFB")

    running = base_value
    bar_bottoms, bar_heights, bar_colors  = [], [], []

    for val in sorted_sv:
        bar_bottoms.append(running if val >= 0 else running + val)
        bar_heights.append(abs(val))
        bar_colors.append("#3B82F6" if val >= 0 else "#EF4444")
        running += val

    y_positions = list(range(n, 0, -1))
    ax.barh(n + 1, base_value, height=0.55, color="#6B7280", alpha=0.7)
    ax.text(base_value / 2, n + 1, f"{base_value:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(-2, n + 1, "Base value", ha="right", va="center", fontsize=9, color="#374151")

    for i, (ypos, name, bottom, height, color, val) in enumerate(zip(y_positions, sorted_names, bar_bottoms, bar_heights, bar_colors, sorted_sv)):
        ax.barh(ypos, height, left=bottom, height=0.55, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        label_x = bottom + height / 2
        sign    = "+" if val >= 0 else ""
        ax.text(label_x, ypos, f"{sign}{val:.1f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        ax.text(-2, ypos, name, ha="right", va="center", fontsize=8.5, color="#374151", linespacing=1.3)

    ax.barh(0, predicted, height=0.55, color="#166534", alpha=0.85)
    ax.text(predicted / 2, 0, f"Score: {predicted:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(-2, 0, "Final score", ha="right", va="center", fontsize=9, color="#374151", fontweight="bold")

    running = base_value
    for i, val in enumerate(sorted_sv):
        end_x = running + val
        if i < n - 1:
            ax.plot([end_x, end_x], [y_positions[i] - 0.28, y_positions[i + 1] + 0.28], color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=5)
        running = end_x

    ax.plot([running, running], [y_positions[-1] - 0.28, 0.28], color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=5)
    ax.set_xlim(-2, 115)
    ax.set_ylim(-0.8, n + 1.8)
    ax.set_yticks([])
    ax.set_xlabel("Compliance Score (0–100)", fontsize=10, color="#374151")
    ax.set_title(f"SHAP Feature Impact  ·  Score: {predicted:.0f}/100", fontsize=13, fontweight="bold", color="#1F4E79", pad=14)

    pos_patch = mpatches.Patch(color="#3B82F6", alpha=0.85, label="Pushes score ↑")
    neg_patch = mpatches.Patch(color="#EF4444", alpha=0.85, label="Pushes score ↓")
    base_patch = mpatches.Patch(color="#6B7280", alpha=0.7, label="Base value")
    ax.legend(handles=[pos_patch, neg_patch, base_patch], loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#E5E7EB")

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.5, zorder=0)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def render_feature_table(features_dict: dict, shap_values: np.ndarray) -> list[dict]:
    rows = []
    for i, fname in enumerate(FEATURE_NAMES):
        sv  = shap_values[i]
        val = features_dict[fname]
        rows.append({
            "Feature":     FEATURE_LABELS.get(fname, fname),
            "Value":       f"{val:.1f}",
            "SHAP Impact": f"{'+' if sv >= 0 else ''}{sv:.2f}",
            "Direction":   "↑ Helps" if sv > 0.5 else ("↓ Hurts" if sv < -0.5 else "→ Neutral"),
        })
    rows.sort(key=lambda r: abs(float(r["SHAP Impact"])), reverse=True)
    return rows