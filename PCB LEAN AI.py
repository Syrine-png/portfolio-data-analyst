import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI + Lean Manufacturing | PCB Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Industrial Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

:root {
    --bg: #0a0c0f;
    --panel: #111418;
    --border: #1e2530;
    --accent: #00d4ff;
    --accent2: #ff6b35;
    --green: #00ff88;
    --red: #ff3355;
    --yellow: #ffd700;
    --orange: #ff9500;
    --violet: #b44fdc;
    --text: #c8d0dc;
    --muted: #4a5568;
    --mono: 'Share Tech Mono', monospace;
    --display: 'Barlow Condensed', sans-serif;
    --body: 'Barlow', sans-serif;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--body); }
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: var(--text) !important; font-family: var(--body); }

h1 { font-family: var(--display) !important; font-weight: 900 !important; font-size: 2.4rem !important;
     letter-spacing: 0.06em !important; color: white !important; text-transform: uppercase; }
h2, h3 { font-family: var(--display) !important; font-weight: 700 !important;
          letter-spacing: 0.04em !important; color: var(--accent) !important; text-transform: uppercase; }
h4 { font-family: var(--mono) !important; color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 0.1em; }

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    padding: 1rem 1.2rem;
    border-radius: 4px;
    font-family: var(--mono);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.7rem !important; letter-spacing: 0.12em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: white !important; font-family: var(--mono) !important; font-size: 1.8rem !important; }
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

.stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
    border-radius: 2px;
}
.stButton > button:hover { background: var(--accent); color: black !important; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

.lean-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.5rem;
    border-radius: 0 4px 4px 0;
    margin: 0.5rem 0;
    font-family: var(--body);
}
.lean-card.warn { border-left-color: var(--accent2); }
.lean-card.ok   { border-left-color: var(--green); }
.lean-card.bad  { border-left-color: var(--red); }
.lean-card.orange { border-left-color: var(--orange); }
.lean-card.violet { border-left-color: var(--violet); }

.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.badge-ok  { background: rgba(0,255,136,0.15); color: var(--green); border: 1px solid var(--green); }
.badge-bad { background: rgba(255,51,85,0.15);  color: var(--red);   border: 1px solid var(--red); }

.section-tag {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    border: 1px solid var(--accent);
    padding: 2px 8px;
    display: inline-block;
    margin-bottom: 0.5rem;
    border-radius: 2px;
    opacity: 0.7;
}

.smed-bar-wrap {
    background: #1a1f2a;
    border-radius: 2px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin: 4px 0 12px;
}
.smed-bar {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.8s ease;
}

.why-step {
    background: var(--panel);
    border-left: 2px solid var(--accent);
    padding: 0.6rem 1rem;
    margin: 4px 0;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--text);
    border-radius: 0 4px 4px 0;
}
.why-arrow { color: var(--accent2); margin-right: 6px; }

.logo-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.75rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.logo-text { font-family: var(--display); font-weight: 900; font-size: 1rem; letter-spacing: 0.12em; color: white; text-transform: uppercase; }
.logo-sub  { font-family: var(--mono); font-size: 0.6rem; color: var(--muted); letter-spacing: 0.15em; }

.batch-progress-wrap {
    background: #1a1f2a;
    border-radius: 3px;
    height: 8px;
    width: 100%;
    margin: 6px 0;
}
.batch-progress-bar {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--green), var(--accent));
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
}
.summary-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent2);
    padding: 1rem;
    text-align: center;
    border-radius: 4px;
}
.summary-val { font-family: var(--mono); font-size: 2rem; color: white; display: block; }
.summary-label { font-family: var(--mono); font-size: 0.6rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }

/* Label badges */
.label-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin: 2px;
}
.badge-mother { background: rgba(255,149,0,0.2); color: var(--orange); border: 1px solid var(--orange); }
.badge-child  { background: rgba(180,79,220,0.2); color: var(--violet); border: 1px solid var(--violet); }

/* Legend row */
.legend-row {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
    font-family: var(--mono);
    font-size: 0.72rem;
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot {
    width: 12px; height: 12px;
    border-radius: 2px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL PATHS  ← adjust to your environment
# ─────────────────────────────────────────────
BASE_DIR         = Path("C:/inspection")
WEIGHTS_DIR      = BASE_DIR / "runs/detect/pcb_v2-5/weights"
CLASSIFY_WEIGHTS = BASE_DIR / "runs/classify/best_Copie.pt"


def find_detect_weights(weights_dir: Path) -> Path | None:
    if not weights_dir.exists():
        return None
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    pt_files = list(weights_dir.glob("*.pt"))
    return pt_files[0] if pt_files else None


DETECT_WEIGHTS = find_detect_weights(WEIGHTS_DIR)

# ─────────────────────────────────────────────
# GOLDEN BOARD  (expected quantities per PCB)
# ─────────────────────────────────────────────
GOLDEN_BOARD = {
    "relay":        32,
    "MOV":          16,
    "Capacitor":    16,
    "Capacitor2":   32,
    "Label Mother":  1,
    "Label Child":  16,
}

# ─────────────────────────────────────────────
# COLOUR MAP for detection boxes
# ─────────────────────────────────────────────
COLOR_MAP = {
    "Label Mother": (0, 165, 255),   # orange (BGR)
    "Label Child":  (180, 0, 180),   # violet (BGR)
    "default":      (255, 255, 0),   # cyan   (BGR)
    "cap_ok":       (0, 255, 0),     # green
    "cap_inv":      (0, 0, 255),     # red
}

CONFIDENCE_THRESHOLD_CLASSIFY = 0.70

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "history":         [],
    "smed_times":      [],
    "run_count":       0,
    "five_why_answers":{},
    "batch_results":   [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def predict_inversion(classify_model, crop):
    import torch
    import torch.nn.functional as F

    crop_r = cv2.resize(crop, (224, 224))
    crop_r = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB) / 255.0
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    crop_r = (crop_r - mean) / std
    tensor = torch.from_numpy(crop_r.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        logits = classify_model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    idx  = int(probs.argmax())
    conf = float(probs[idx])

    if conf < CONFIDENCE_THRESHOLD_CLASSIFY:
        return "OK", conf
    return ["Inverted", "OK"][idx], conf


@st.cache_resource
def load_models():
    import torch
    import timm
    from ultralytics import YOLO

    detect_model = YOLO(str(DETECT_WEIGHTS))

    state_dict = torch.load(str(CLASSIFY_WEIGHTS), map_location="cpu")
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("model",
                     state_dict.get("state_dict", state_dict))

    classify_model = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=2
    )
    classify_model.load_state_dict(state_dict)
    return detect_model, classify_model.eval()


def inspect_single_image(img_bytes, filename, detect_model, classify_model):
    """Full inspection pipeline → (result_dict, annotated_img, crops_list)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_bytes)
        path = tmp.name

    img = cv2.imread(path)
    if img is None:
        os.unlink(path)
        return None, None, []

    t_start = time.time()
    results = detect_model.predict(path, conf=0.5)
    t_end   = time.time()
    proc_time = round(t_end - t_start, 3)

    boxes         = results[0].boxes
    img_draw      = img.copy()
    detected_counts = {}
    defect_list   = []
    inverted_caps = 0
    crops_display = []            # (crop_bgr, caption_text)

    for box in boxes:
        cls        = int(box.cls[0])
        label      = results[0].names[cls]
        conf_score = float(box.conf[0])
        detected_counts[label] = detected_counts.get(label, 0) + 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── Capacitors: polarity check ──
        if label.lower() in ["capacitor", "capacitor2"]:
            crop = img[y1:y2, x1:x2]
            if crop.size != 0:
                polarity, pol_conf = predict_inversion(classify_model, crop)
                color = COLOR_MAP["cap_ok"] if polarity == "OK" else COLOR_MAP["cap_inv"]
                text  = f"{label} — {polarity} ({pol_conf:.2f})"
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                crops_display.append((crop.copy(), text))
                if polarity == "Inverted":
                    inverted_caps += 1
                    defect_list.append(f"{label} Inversé")

        # ── Label Mother ──
        elif label == "Label Mother":
            color = COLOR_MAP["Label Mother"]
            text  = f"MOTHER ({conf_score:.2f})"
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_draw, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── Label Child ──
        elif label == "Label Child":
            color = COLOR_MAP["Label Child"]
            text  = f"CHILD #{detected_counts[label]} ({conf_score:.2f})"
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ── Other components ──
        else:
            color = COLOR_MAP["default"]
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_draw, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Component analysis vs Golden Board ──
    component_analysis = []
    for comp, expected in GOLDEN_BOARD.items():
        detected = detected_counts.get(comp, 0)
        if detected == expected:
            decision = "✅ OK"
        elif detected < expected:
            decision = f"❌ {expected - detected} manquant(s)"
            defect_list.append(f"{comp} Manquant")
        else:
            decision = f"⚠️ {detected - expected} surplus"
            defect_list.append(f"{comp} Surplus")
        component_analysis.append({
            "Composant": comp,
            "Attendu":   expected,
            "Détecté":   detected,
            "Décision":  decision,
        })

    # ── Final verdict ──
    has_issue     = any(r["Décision"] != "✅ OK" for r in component_analysis)
    final_decision = "DEFECTIVE" if (has_issue or inverted_caps > 0) else "PASS"

    os.unlink(path)

    result = {
        "Filename":    filename,
        "Run":         st.session_state.run_count + 1,
        "Time":        proc_time,
        "Decision":    final_decision,
        "Defects":     defect_list,
        "FPY":         100.0 if final_decision == "PASS" else 0.0,
        "Timestamp":   datetime.now().strftime("%H:%M:%S"),
        "Components":  component_analysis,
        "Inverted":    inverted_caps,
        "LabelMother": detected_counts.get("Label Mother", 0),
        "LabelChild":  detected_counts.get("Label Child", 0),
    }
    return result, img_draw, crops_display


def compute_kpis(history):
    if not history:
        return 0.0, 100.0, 0.0, 88.0
    total      = len(history)
    defective  = sum(1 for h in history if h["Decision"] == "DEFECTIVE")
    defect_rate = (defective / total) * 100
    fpy         = 100 - defect_rate
    oee         = min(100, 88 + (fpy - 95) * 0.5) if fpy >= 50 else 60.0
    avg_time    = float(np.mean([h["Time"] for h in history]))
    return defect_rate, fpy, avg_time, oee


def pareto_data(history):
    defect_types = {}
    for h in history:
        for d in h.get("Defects", []):
            defect_types[d] = defect_types.get(d, 0) + 1
    return dict(sorted(defect_types.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-bar">
        <div>
            <div class="logo-text">🏭 PCB Lean AI</div>
            <div class="logo-sub">v4.0 · Labels + Batch · Industrial Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.selectbox(
        "MODULE",
        ["🔬 Inspection (Batch)", "📊 Lean Dashboard", "📈 Kaizen Tracking",
         "⏱️ SMED Analysis", "📉 Pareto Analysis", "❓ 5 Why Analysis"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown('<div class="section-tag">System Status</div>', unsafe_allow_html=True)

    model_ok = (DETECT_WEIGHTS is not None and DETECT_WEIGHTS.exists()
                and CLASSIFY_WEIGHTS.exists())
    if model_ok:
        st.markdown('<span class="status-badge badge-ok">● MODELS READY</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-bad">● MODELS NOT FOUND</span>',
                    unsafe_allow_html=True)
        st.caption(f"Detection  : `{DETECT_WEIGHTS}`")
        st.caption(f"Classify   : `{CLASSIFY_WEIGHTS}`")

    st.markdown("---")
    st.markdown('<div class="section-tag">Golden Board</div>', unsafe_allow_html=True)
    for comp, qty in GOLDEN_BOARD.items():
        badge_cls = "badge-mother" if comp == "Label Mother" \
                    else "badge-child" if comp == "Label Child" else ""
        if badge_cls:
            st.markdown(
                f'<span class="label-badge {badge_cls}">{comp}: {qty}</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span style="font-family:var(--mono);font-size:0.72rem;'
                f'color:var(--text)">{comp}: <b style="color:var(--accent)">{qty}</b></span>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<div class="section-tag">Session Stats</div>', unsafe_allow_html=True)

    total_insp   = len(st.session_state.history)
    total_defect = sum(1 for h in st.session_state.history if h["Decision"] == "DEFECTIVE")
    fpy_glob     = round((1 - total_defect / total_insp) * 100, 1) if total_insp else 100.0

    st.markdown(f"""
    <p style="font-family:var(--mono);font-size:0.75rem;color:var(--text)">
        PCBs inspectés: <b style="color:var(--accent)">{total_insp}</b><br>
        Défectueux: <b style="color:var(--red)">{total_defect}</b><br>
        FPY global: <b style="color:var(--green)">{fpy_glob}%</b>
    </p>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Reset Session"):
        for k in ["history", "smed_times", "batch_results", "five_why_answers"]:
            st.session_state[k] = [] if k != "five_why_answers" else {}
        st.session_state.run_count = 0
        st.rerun()


# ═════════════════════════════════════════════
# MODULE 1 — INSPECTION (BATCH)
# ═════════════════════════════════════════════
if menu == "🔬 Inspection (Batch)":
    st.markdown("## 🔬 PCB Batch Inspection")
    st.markdown(
        '<div class="section-tag">AI · YOLO + ViT · Multi-Image · Label Tracking</div>',
        unsafe_allow_html=True
    )

    # Legend
    st.markdown("""
    <div class="legend-row">
        <span class="legend-item"><span class="legend-dot" style="background:#ff9500"></span>Label Mother</span>
        <span class="legend-item"><span class="legend-dot" style="background:#b44fdc"></span>Label Child</span>
        <span class="legend-item"><span class="legend-dot" style="background:#00ff88"></span>Capacitor OK</span>
        <span class="legend-item"><span class="legend-dot" style="background:#ff3355"></span>Capacitor Inversé</span>
        <span class="legend-item"><span class="legend-dot" style="background:#ffff00"></span>Autres composants</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Sélectionner les images PCB",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        n = len(uploaded_files)
        st.markdown(
            f'<div class="lean-card"><b>{n} image(s) sélectionnée(s)</b> — Prêt pour l\'analyse batch.</div>',
            unsafe_allow_html=True
        )

        if not model_ok:
            st.error("⚠️ Modèles introuvables. Vérifiez les chemins.")
        else:
            if st.button(f"▶ Lancer l'inspection batch ({n} PCB)"):
                detect_model, classify_model = load_models()

                batch_results  = []
                annotated_imgs = []
                all_crops      = []    # list of (crop, caption) for display
                progress_ph    = st.empty()
                status_ph      = st.empty()

                for i, uf in enumerate(uploaded_files):
                    pct = int((i / n) * 100)
                    progress_ph.markdown(f"""
                    <div style="margin:0.5rem 0">
                        <div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-bottom:4px">
                            Traitement : {uf.name} ({i+1}/{n})
                        </div>
                        <div class="batch-progress-wrap">
                            <div class="batch-progress-bar" style="width:{pct}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    status_ph.markdown(
                        '<span style="font-family:var(--mono);font-size:0.7rem;'
                        'color:var(--accent)">⚙ Analyse en cours…</span>',
                        unsafe_allow_html=True
                    )

                    result, img_draw, crops = inspect_single_image(
                        uf.read(), uf.name, detect_model, classify_model
                    )
                    if result:
                        batch_results.append(result)
                        annotated_imgs.append(img_draw)
                        all_crops.append(crops)
                        st.session_state.history.append(result)
                        st.session_state.smed_times.append(result["Time"])
                        st.session_state.run_count += 1

                progress_ph.markdown("""
                <div class="batch-progress-wrap">
                    <div class="batch-progress-bar" style="width:100%;background:var(--green)"></div>
                </div>
                """, unsafe_allow_html=True)
                status_ph.markdown(
                    '<span style="font-family:var(--mono);font-size:0.7rem;'
                    'color:var(--green)">✅ Batch terminé</span>',
                    unsafe_allow_html=True
                )
                st.session_state.batch_results = batch_results

                # ── Batch Summary ──
                st.markdown("---")
                st.markdown("### 📊 Résumé Batch")

                total_b  = len(batch_results)
                pass_b   = sum(1 for r in batch_results if r["Decision"] == "PASS")
                defect_b = total_b - pass_b
                fpy_b    = round(pass_b / total_b * 100, 1) if total_b else 0
                avg_t_b  = round(float(np.mean([r["Time"] for r in batch_results])), 3)

                st.markdown(f"""
                <div class="summary-grid">
                    <div class="summary-box">
                        <span class="summary-val">{total_b}</span>
                        <span class="summary-label">PCBs analysés</span>
                    </div>
                    <div class="summary-box" style="border-top-color:var(--green)">
                        <span class="summary-val" style="color:var(--green)">{pass_b}</span>
                        <span class="summary-label">PASS</span>
                    </div>
                    <div class="summary-box" style="border-top-color:var(--red)">
                        <span class="summary-val" style="color:var(--red)">{defect_b}</span>
                        <span class="summary-label">DEFECTIVE</span>
                    </div>
                    <div class="summary-box" style="border-top-color:var(--accent)">
                        <span class="summary-val">{fpy_b}%</span>
                        <span class="summary-label">First Pass Yield</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if fpy_b >= 95:
                    v_cls, v_txt = "ok",   f"✅ Excellent FPY ({fpy_b}%) — Objectif 95% atteint."
                elif fpy_b >= 80:
                    v_cls, v_txt = "warn", f"⚠️ FPY acceptable ({fpy_b}%) — Améliorations Kaizen recommandées."
                else:
                    v_cls, v_txt = "bad",  f"❌ FPY critique ({fpy_b}%) — Action corrective immédiate requise."
                st.markdown(f'<div class="lean-card {v_cls}"><b>{v_txt}</b></div>',
                            unsafe_allow_html=True)

                # ── Label summary ──
                st.markdown("### 🏷️ Résumé Étiquettes (batch)")
                total_mothers = sum(r["LabelMother"] for r in batch_results)
                total_children = sum(r["LabelChild"] for r in batch_results)
                expected_mothers = total_b * GOLDEN_BOARD["Label Mother"]
                expected_children = total_b * GOLDEN_BOARD["Label Child"]

                col_m, col_c = st.columns(2)
                with col_m:
                    if total_mothers == expected_mothers:
                        st.success(f"✅ Label Mother — {total_mothers}/{expected_mothers} détectées")
                    else:
                        delta = expected_mothers - total_mothers
                        st.error(f"❌ Label Mother — {total_mothers}/{expected_mothers} — {abs(delta)} {'manquante(s)' if delta > 0 else 'en surplus'}")
                with col_c:
                    if total_children == expected_children:
                        st.success(f"✅ Label Child — {total_children}/{expected_children} détectées")
                    else:
                        delta = expected_children - total_children
                        st.error(f"❌ Label Child — {total_children}/{expected_children} — {abs(delta)} {'manquante(s)' if delta > 0 else 'en surplus'}")

                # ── Per-PCB results table ──
                st.markdown("### 📋 Résultats par PCB")
                df_batch = pd.DataFrame([{
                    "Fichier":     r["Filename"],
                    "Décision":    r["Decision"],
                    "Défauts":     ", ".join(r["Defects"]) if r["Defects"] else "—",
                    "Inversions":  r["Inverted"],
                    "L.Mother":    r["LabelMother"],
                    "L.Child":     r["LabelChild"],
                    "Temps (s)":   r["Time"],
                    "Heure":       r["Timestamp"],
                } for r in batch_results])
                st.dataframe(df_batch, use_container_width=True)

                # ── Defect frequency ──
                all_defects = [d for r in batch_results for d in r["Defects"]]
                if all_defects:
                    st.markdown("### 📉 Fréquence des défauts (aperçu Pareto)")
                    defect_counts = Counter(all_defects)
                    df_def = pd.DataFrame(defect_counts.items(),
                                          columns=["Défaut", "Occurrences"])
                    df_def = df_def.sort_values("Occurrences", ascending=False).reset_index(drop=True)
                    df_def["% cumulé"] = (
                        df_def["Occurrences"].cumsum() / df_def["Occurrences"].sum() * 100
                    ).round(1)
                    st.dataframe(df_def, use_container_width=True)
                    st.bar_chart(df_def.set_index("Défaut")["Occurrences"])

                # ── Annotated images grid ──
                st.markdown("### 🖼️ Images annotées")
                cols_per_row = 3
                for row_start in range(0, len(annotated_imgs), cols_per_row):
                    row_imgs = list(zip(
                        annotated_imgs[row_start:row_start + cols_per_row],
                        batch_results[row_start:row_start + cols_per_row],
                        all_crops[row_start:row_start + cols_per_row],
                    ))
                    cols = st.columns(len(row_imgs))
                    for col, (img_draw, res, crops) in zip(cols, row_imgs):
                        with col:
                            badge = "✅" if res["Decision"] == "PASS" else "❌"
                            st.image(
                                cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                                caption=f"{badge} {res['Filename']}"
                            )
                            # Show label counts inline
                            m_col = "🟠" if res["LabelMother"] == GOLDEN_BOARD["Label Mother"] else "🔴"
                            c_col = "🟣" if res["LabelChild"] == GOLDEN_BOARD["Label Child"] else "🔴"
                            st.caption(
                                f"{m_col} Mother: {res['LabelMother']}/1  "
                                f"{c_col} Child: {res['LabelChild']}/16"
                            )

                # ── Capacitor crops ──
                all_caps = [(crop, cap) for crops in all_crops for crop, cap in crops]
                if all_caps:
                    st.markdown("### 🔍 Crops Capacitors (polarité)")
                    cap_cols = st.columns(4)
                    for i, (crop, caption) in enumerate(all_caps[:12]):   # max 12 shown
                        with cap_cols[i % 4]:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                     caption=caption, use_container_width=True)

    elif st.session_state.batch_results:
        st.markdown("### 📋 Derniers résultats batch (session courante)")
        df_b = pd.DataFrame([{
            "Fichier":   r["Filename"],
            "Décision":  r["Decision"],
            "Défauts":   ", ".join(r["Defects"]) if r["Defects"] else "—",
            "Temps (s)": r["Time"],
        } for r in st.session_state.batch_results])
        st.dataframe(df_b, use_container_width=True)
    else:
        st.markdown(
            '<div class="lean-card warn">⬆️ Uploadez vos images PCB pour démarrer l\'analyse batch.</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════
# MODULE 2 — LEAN DASHBOARD
# ═════════════════════════════════════════════
elif menu == "📊 Lean Dashboard":
    st.markdown("## 📊 Lean KPI Dashboard")
    st.markdown(
        '<div class="section-tag">OEE · FPY · Defect Rate · SMED · Kaizen</div>',
        unsafe_allow_html=True
    )

    defect_rate, fpy, avg_time, oee = compute_kpis(st.session_state.history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔴 Defect Rate", f"{defect_rate:.1f}%",
                  delta=f"{-defect_rate:.1f}% vs target", delta_color="inverse")
    with c2:
        st.metric("✅ First Pass Yield", f"{fpy:.1f}%",
                  delta=f"+{fpy - 95:.1f}% vs 95% target")
    with c3:
        st.metric("⚙️ OEE (Simulé)", f"{oee:.1f}%",
                  delta=f"{oee - 85:.1f}% vs 85% baseline")
    with c4:
        st.metric("⏱️ Avg Process Time", f"{avg_time:.2f}s",
                  delta="SMED target: <1.0s")

    # Label summary across full session
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🏷️ Suivi Étiquettes — Session")
        n_sess      = len(st.session_state.history)
        tot_mother  = sum(h.get("LabelMother", 0) for h in st.session_state.history)
        tot_child   = sum(h.get("LabelChild",  0) for h in st.session_state.history)
        exp_mother  = n_sess * GOLDEN_BOARD["Label Mother"]
        exp_child   = n_sess * GOLDEN_BOARD["Label Child"]

        lm1, lm2 = st.columns(2)
        with lm1:
            st.markdown(f"""
            <div class="lean-card {'ok' if tot_mother == exp_mother else 'bad'}">
                <span class="label-badge badge-mother">MOTHER</span><br>
                Détectées : <b>{tot_mother}</b> / Attendues : <b>{exp_mother}</b>
            </div>""", unsafe_allow_html=True)
        with lm2:
            st.markdown(f"""
            <div class="lean-card {'ok' if tot_child == exp_child else 'bad'}">
                <span class="label-badge badge-child">CHILD</span><br>
                Détectées : <b>{tot_child}</b> / Attendues : <b>{exp_child}</b>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Historique des inspections")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)[
            ["Filename", "Run", "Timestamp", "Time", "Decision", "FPY",
             "LabelMother", "LabelChild", "Inverted"]
        ]
        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv, "pcb_results.csv", "text/csv")
    else:
        st.markdown(
            '<div class="lean-card warn">Aucune donnée. Lancez un batch d\'abord.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🏭 Principes Lean appliqués")
    cols = st.columns(3)
    principles = [
        ("♻️ Kaizen",  "Amélioration continue via FPY, taux de défauts et suivi des étiquettes."),
        ("⏱️ SMED",    "Temps d'analyse minimisé. Objectif sub-seconde par PCB."),
        ("📉 Pareto",  "Causes de défauts classées 80/20 pour concentrer les actions."),
    ]
    for i, (title, desc) in enumerate(principles):
        with cols[i]:
            st.markdown(
                f'<div class="lean-card"><b>{title}</b><br>'
                f'<span style="font-size:0.85rem;opacity:0.8">{desc}</span></div>',
                unsafe_allow_html=True
            )


# ═════════════════════════════════════════════
# MODULE 3 — KAIZEN
# ═════════════════════════════════════════════
elif menu == "📈 Kaizen Tracking":
    st.markdown("## 📈 Kaizen — Amélioration Continue")
    st.markdown(
        '<div class="section-tag">Quality Trend · FPY Evolution · Defect Reduction</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) < 2:
        st.markdown(
            '<div class="lean-card warn">⚠️ Analysez au moins 2 PCBs pour visualiser les tendances Kaizen.</div>',
            unsafe_allow_html=True
        )
    else:
        df_k = pd.DataFrame(st.session_state.history)
        df_k["Defect Rate"]         = df_k["Decision"].apply(lambda d: 100 if d == "DEFECTIVE" else 0)
        df_k["Defect Rate Rolling"] = df_k["Defect Rate"].expanding().mean()
        df_k["FPY Rolling"]         = df_k["FPY"].expanding().mean()
        df_k["Label Mother Rate"]   = df_k.get("LabelMother", 0)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Taux de défauts (↓ = Kaizen efficace)")
            st.line_chart(df_k[["Defect Rate Rolling"]].rename(
                columns={"Defect Rate Rolling": "Defect Rate %"}
            ))
        with col_b:
            st.markdown("#### First Pass Yield (↑ = Kaizen efficace)")
            st.line_chart(df_k[["FPY Rolling"]].rename(
                columns={"FPY Rolling": "FPY %"}
            ))

        last_dr  = df_k["Defect Rate Rolling"].iloc[-1]
        first_dr = df_k["Defect Rate Rolling"].iloc[0]
        trend    = "↓ En amélioration" if last_dr <= first_dr else "↑ En dégradation"
        badge    = "ok" if last_dr <= first_dr else "bad"
        st.markdown(
            f'<div class="lean-card {badge}"><b>Verdict Kaizen :</b> '
            f'Taux de défauts <b>{trend}</b>. Moyenne courante : <b>{last_dr:.1f}%</b></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🧠 Cycle PDCA")
    for step, desc in [
        ("PLAN",  "Identifier les patterns de défauts (étiquettes manquantes, inversions) depuis les données batch"),
        ("DO",    "Ajuster seuils de détection, ré-entraîner le modèle ViT ou YOLO"),
        ("CHECK", "Comparer taux de défauts avant vs après ajustement"),
        ("ACT",   "Standardiser les améliorations, mettre à jour le golden board si nécessaire"),
    ]:
        st.markdown(
            f'<div class="lean-card"><span style="color:var(--accent);font-family:var(--mono);'
            f'font-size:0.8rem">[{step}]</span> &nbsp;{desc}</div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════
# MODULE 4 — SMED
# ═════════════════════════════════════════════
elif menu == "⏱️ SMED Analysis":
    st.markdown("## ⏱️ SMED — Single Minute Exchange of Die")
    st.markdown(
        '<div class="section-tag">Processing Time Reduction · Throughput Optimization</div>',
        unsafe_allow_html=True
    )

    times  = st.session_state.smed_times
    TARGET = 1.0

    c1, c2, c3, c4 = st.columns(4)
    if times:
        avg_t   = float(np.mean(times))
        best_t  = float(np.min(times))
        worst_t = float(np.max(times))
        under   = sum(1 for t in times if t < TARGET)
    else:
        avg_t = best_t = worst_t = 0.0
        under = 0

    with c1: st.metric("⌀ Temps moyen",     f"{avg_t:.3f}s")
    with c2: st.metric("🏆 Meilleur temps",  f"{best_t:.3f}s")
    with c3: st.metric("⚠️ Pire temps",      f"{worst_t:.3f}s")
    with c4: st.metric("✅ < 1s (objectif)", f"{under}/{len(times)}" if times else "0/0")

    if times:
        latest = times[-1]
        pct    = min(100, (latest / TARGET) * 100)
        color  = "var(--green)" if latest < TARGET else "var(--red)"
        st.markdown(f"""
        <div style='margin:1rem 0'>
            <div style='font-family:var(--mono);font-size:0.7rem;color:var(--muted);margin-bottom:4px'>
                DERNIER RUN : {latest:.3f}s vs OBJECTIF : {TARGET}s
            </div>
            <div class='smed-bar-wrap'>
                <div class='smed-bar' style='width:{pct}%;background:{color}'></div>
            </div>
            <div style='font-family:var(--mono);font-size:0.7rem;color:{color}'>
                {'✅ SOUS OBJECTIF' if latest < TARGET else '⚠️ DÉPASSE OBJECTIF'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Temps de traitement par PCB")
        df_smed = pd.DataFrame({"PCB #": range(1, len(times) + 1), "Temps (s)": times})
        df_smed["Objectif"] = TARGET
        st.line_chart(df_smed.set_index("PCB #"))

        st.markdown("#### Distribution des temps")
        bins   = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
        labels = ["<0.5s", "0.5–1s", "1–1.5s", "1.5–2s", ">2s"]
        counts, _ = np.histogram(times, bins=bins)
        st.bar_chart(pd.DataFrame({"Tranche": labels, "Nombre": counts}).set_index("Tranche"))
    else:
        st.markdown(
            '<div class="lean-card warn">Aucune donnée. Lancez un batch d\'inspection.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔧 Leviers d'optimisation SMED")
    for lever, detail in [
        ("Quantisation du modèle",   "Réduire YOLO/ViT en INT8 → inférence plus rapide"),
        ("Traitement batch GPU",     "Grouper plusieurs PCBs par appel CUDA"),
        ("Résolution adaptative",    "Réduire résolution d'entrée si précision le permet"),
        ("Pipeline préchargé",       "@st.cache_resource : modèles en mémoire (déjà actif)"),
        ("Seuil de confiance ViT",   f"Threshold actuel : {CONFIDENCE_THRESHOLD_CLASSIFY} — ajustable"),
        ("Parallélisme multi-thread","ThreadPoolExecutor pour inspecter plusieurs images en simultané"),
    ]:
        st.markdown(
            f'<div class="lean-card"><b style="color:var(--accent)">{lever}</b><br>'
            f'<span style="font-size:0.85rem">{detail}</span></div>',
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════
# MODULE 5 — PARETO
# ═════════════════════════════════════════════
elif menu == "📉 Pareto Analysis":
    st.markdown("## 📉 Analyse Pareto — Règle 80/20")
    st.markdown(
        '<div class="section-tag">Fréquence des défauts · Priorisation des causes racines</div>',
        unsafe_allow_html=True
    )

    pd_data = pareto_data(st.session_state.history)

    if not pd_data:
        st.markdown(
            '<div class="lean-card warn">Aucun défaut collecté. Analysez un batch de PCBs.</div>',
            unsafe_allow_html=True
        )
    else:
        df_p = pd.DataFrame(list(pd_data.items()), columns=["Type de défaut", "Occurrences"])
        df_p = df_p.sort_values("Occurrences", ascending=False).reset_index(drop=True)
        total = df_p["Occurrences"].sum()
        df_p["% individuel"] = (df_p["Occurrences"] / total * 100).round(1)
        df_p["% cumulé"]     = (df_p["Occurrences"].cumsum() / total * 100).round(1)

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.markdown("#### Fréquence des défauts")
            st.bar_chart(df_p.set_index("Type de défaut")["Occurrences"])
        with col_table:
            st.markdown("#### Tableau Pareto")
            st.dataframe(df_p, use_container_width=True)

        vital_few = df_p[df_p["% cumulé"] <= 80]
        if not vital_few.empty:
            st.markdown(
                f'<div class="lean-card ok"><b>Vital Few (80%) :</b> Concentrez-vous sur '
                f'<b>{", ".join(vital_few["Type de défaut"].tolist())}</b> — '
                f'ces défauts représentent 80% des non-conformités.</div>',
                unsafe_allow_html=True
            )
        else:
            top1 = df_p.iloc[0]["Type de défaut"]
            st.markdown(
                f'<div class="lean-card ok"><b>Défaut prioritaire :</b> '
                f'<b>{top1}</b> ({df_p.iloc[0]["% individuel"]}% des cas).</div>',
                unsafe_allow_html=True
            )

        # Label-specific stats
        label_defects = {k: v for k, v in pd_data.items()
                         if "Mother" in k or "Child" in k}
        if label_defects:
            st.markdown("---")
            st.markdown("#### 🏷️ Focus étiquettes")
            for lbl, cnt in label_defects.items():
                badge_cls = "badge-mother" if "Mother" in lbl else "badge-child"
                st.markdown(
                    f'<div class="lean-card orange"><span class="label-badge {badge_cls}">{lbl}</span> '
                    f'— <b>{cnt}</b> occurrence(s) sur {len(st.session_state.history)} PCB(s)</div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("""
    <div class="lean-card">
    <b>80/20 en inspection PCB :</b><br>
    En général, 20% des types de défauts causent 80% des cartes rejetées.<br>
    Priorisez les actions sur les 1–2 types identifiés ci-dessus (inversions, étiquettes manquantes…).
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODULE 6 — 5 WHY
# ═════════════════════════════════════════════
elif menu == "❓ 5 Why Analysis":
    st.markdown("## ❓ Analyse 5 Pourquoi — Investigation Cause Racine")
    st.markdown(
        '<div class="section-tag">Root Cause · Corrective Action · Problem Solving</div>',
        unsafe_allow_html=True
    )

    pd_data = pareto_data(st.session_state.history)
    default_problem = list(pd_data.keys())[0] if pd_data else "Capacitor Inversé"

    problem = st.text_input(
        "🔍 Définir le problème :",
        value=f"Défaut détecté : {default_problem}"
    )

    st.markdown("### Décomposition — Poser 5 fois POURQUOI")

    why_defaults = {
        1: ("Pourquoi le PCB est-il défectueux ?",
            "Parce qu'un composant est manquant ou inversé."),
        2: ("Pourquoi le composant est-il absent/inversé ?",
            "Parce que la vérification manuelle était insuffisante."),
        3: ("Pourquoi la vérification était-elle insuffisante ?",
            "Parce qu'aucune vérification automatique n'existait avant ce système IA."),
        4: ("Pourquoi n'y avait-il pas de vérification automatique ?",
            "Parce que le processus reposait uniquement sur l'inspection visuelle humaine."),
        5: ("Pourquoi seulement l'inspection manuelle ?",
            "Parce que l'investissement dans l'IA industrielle n'avait pas été priorisé."),
    }

    for i in range(1, 6):
        q_default, a_default = why_defaults[i]
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:0.7rem;'
            f'color:var(--accent);margin-top:1rem">POURQUOI #{i}</div>',
            unsafe_allow_html=True
        )
        st.text_input(f"Question #{i}", value=q_default, key=f"why_q_{i}")
        why_a = st.text_input(f"Réponse #{i}", value=a_default, key=f"why_a_{i}")
        st.markdown(
            f'<div class="why-step"><span class="why-arrow">→</span>{why_a}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🎯 Cause racine & Action corrective")

    col_l, col_r = st.columns(2)
    with col_l:
        root_cause = st.text_area(
            "Cause racine identifiée :",
            value="Sur-dépendance à l'inspection manuelle → erreurs non détectées "
                  "(polarité inversée, étiquettes absentes).",
            height=100
        )
    with col_r:
        corrective = st.text_area(
            "Action corrective :",
            value="Déployer ce système IA sur la ligne de production. "
                  "Ré-entraîner le modèle ViT & YOLO trimestriellement "
                  "avec de nouveaux échantillons (labels inclus).",
            height=100
        )

    if st.button("💾 Sauvegarder le rapport 5 Why"):
        st.session_state.five_why_answers = {
            "problem":    problem,
            "root_cause": root_cause,
            "corrective": corrective,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.success("✅ Rapport sauvegardé en session.")

    if st.session_state.five_why_answers:
        st.markdown("---")
        st.markdown("#### 📄 Rapport sauvegardé")
        r = st.session_state.five_why_answers
        st.markdown(f"""
        <div class="lean-card ok">
            <b>Problème :</b> {r['problem']}<br>
            <b>Cause racine :</b> {r['root_cause']}<br>
            <b>Action :</b> {r['corrective']}<br>
            <span style="font-size:0.7rem;color:var(--muted)">Enregistré le : {r['timestamp']}</span>
        </div>
        """, unsafe_allow_html=True)