import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import io
import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Inspection System",
    page_icon="🔬",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; padding: 0.5rem 1.5rem; }
    .stTabs [aria-selected="true"] { color: #4A9EFF !important; border-bottom: 3px solid #4A9EFF !important; }
    .cam-live  { color: #28a745; font-weight: 700; font-size: 1rem; }
    .cam-off   { color: #dc3545; font-weight: 700; font-size: 1rem; }
    .export-section {
        background: #1a1a2e;
        border: 1px solid #4A9EFF44;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHEMINS MODÈLES
# ─────────────────────────────────────────────
BASE_DIR        = Path("C:/inspection")
WEIGHTS_DIR     = BASE_DIR / "runs\detect\pcb_v2-5\weights"
CLASSIFY_WEIGHTS = BASE_DIR / "runs/classify/best_Copie.pt"

def find_detect_weights(weights_dir: Path) -> Path:
    if not weights_dir.exists():
        return None
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    pt_files = list(weights_dir.glob("*.pt"))
    return pt_files[0] if pt_files else None

DETECT_WEIGHTS = find_detect_weights(WEIGHTS_DIR)

# ─────────────────────────────────────────────
# GOLDEN BOARD
# ─────────────────────────────────────────────
golden_board = {
    "relay":        32,
    "MOV":          16,
    "Capacitor":    16,
    "Capacitor2":   32,
    "Label Mother":  1,
    "Label Child":  16,
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "captured_frame": None,
    "show_analysis":  False,
    "cam_index":      1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# CLASSIFICATION (ViT)
# ─────────────────────────────────────────────
def predict_inversion(classify_model, crop):
    import torch
    import torch.nn.functional as F

    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    crop = (crop - mean) / std
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits = classify_model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    idx  = int(probs.argmax())
    conf = float(probs[idx])
    classes = ["Inverted", "OK"]
    return ("OK", conf) if conf < 0.70 else (classes[idx], conf)

# ─────────────────────────────────────────────
# CHARGEMENT MODÈLES
# ─────────────────────────────────────────────
@st.cache_resource
def load_models(detect_path: str, classify_path: str):
    import torch, timm
    from ultralytics import YOLO

    detect_model = YOLO(detect_path)
    state_dict   = torch.load(classify_path, map_location="cpu")
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))
    classify_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    classify_model.load_state_dict(state_dict)
    classify_model.eval()
    return detect_model, classify_model

# ─────────────────────────────────────────────
# EXPORT CSV
# ─────────────────────────────────────────────
def build_csv(table_data, label_mother, label_child, verdict, timestamp):
    """Génère un CSV complet avec en-tête de session."""
    rows = []
    # En-tête de session
    rows.append(["Rapport d'inspection PCB"])
    rows.append(["Date/Heure", timestamp])
    rows.append(["Décision finale", verdict])
    rows.append([])
    # Tableau composants
    rows.append(["Composant", "Quantité attendue", "Quantité détectée", "Décision"])
    for row in table_data:
        # Enlever les emojis pour le CSV
        clean_row = [str(c).replace("✅","OK").replace("❌","NOK").replace("⚠️","WARN") for c in row]
        rows.append(clean_row)
    rows.append([])
    # Labels
    rows.append(["Résumé étiquettes"])
    rows.append(["Label Mother", f"{label_mother}/1"])
    rows.append(["Label Child",  f"{label_child}/16"])

    buf = io.StringIO()
    import csv
    writer = csv.writer(buf, delimiter=";")
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")   # BOM pour Excel


# ─────────────────────────────────────────────
# EXPORT PDF  (fpdf2)
# ─────────────────────────────────────────────
def build_pdf(table_data, label_mother, label_child, verdict, timestamp, img_draw_rgb=None):
    """Génère un PDF de rapport avec tableau et image annotée."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None  # fpdf2 non installé → bouton désactivé

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(30, 80, 200)
            self.cell(0, 10, "Rapport d'Inspection PCB — Zollner Elektronik Tunisie", ln=True, align="C")
            self.set_text_color(0, 0, 0)
            self.set_font("Helvetica", "", 9)
            self.cell(0, 6, f"Généré le : {timestamp}", ln=True, align="C")
            self.ln(3)
            self.set_draw_color(74, 158, 255)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Verdict ──
    ok = "CONFORME" in verdict.upper() or "OK" in verdict.upper()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(220, 255, 220) if ok else pdf.set_fill_color(255, 210, 210)
    pdf.set_text_color(0, 128, 0) if ok else pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 10, f"Décision finale : {'✓ PCB CONFORME' if ok else '✗ PCB DEFECTUEUX'}", ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Image annotée ──
    if img_draw_rgb is not None:
        tmp_img = "/tmp/_pcb_report_img.jpg"
        cv2.imwrite(tmp_img, cv2.cvtColor(img_draw_rgb, cv2.COLOR_RGB2BGR))
        # Largeur max 180 mm, hauteur auto
        img_h, img_w = img_draw_rgb.shape[:2]
        disp_w = 180
        disp_h = int(disp_w * img_h / img_w)
        if disp_h > 110:           # plafond en hauteur
            disp_h = 110
            disp_w = int(disp_h * img_w / img_h)
        pdf.image(tmp_img, x=(210 - disp_w) / 2, w=disp_w, h=disp_h)
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "Image annotée par le système de vision", ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # ── Tableau composants ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Analyse des composants", ln=True)
    pdf.ln(1)

    col_w = [50, 45, 45, 50]
    headers = ["Composant", "Attendu", "Détecté", "Statut"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(74, 158, 255)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 8, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for i, row in enumerate(table_data):
        # Couleur de fond selon statut
        decision = str(row[3])
        if "OK" in decision:
            pdf.set_fill_color(235, 255, 235)
        elif "manquant" in decision or "❌" in decision:
            pdf.set_fill_color(255, 225, 225)
        else:
            pdf.set_fill_color(255, 245, 215)
        pdf.set_text_color(0, 0, 0)
        fill = True
        clean = [str(c).replace("✅","OK").replace("❌","NOK").replace("⚠️","WARN") for c in row]
        for val, w in zip(clean, col_w):
            pdf.cell(w, 7, val, border=1, fill=fill, align="C")
        pdf.ln()

    pdf.ln(5)

    # ── Résumé labels ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Résumé des étiquettes", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(235, 235, 255)
    pdf.set_text_color(0, 0, 0)
    lm_ok = label_mother == 1
    lc_ok = label_child == 16
    pdf.set_fill_color(235, 255, 235) if lm_ok else pdf.set_fill_color(255, 225, 225)
    pdf.cell(95, 7, f"Label Mother : {label_mother}/1  ({'OK' if lm_ok else 'NOK'})", border=1, fill=True, align="C")
    pdf.set_fill_color(235, 255, 235) if lc_ok else pdf.set_fill_color(255, 225, 225)
    pdf.cell(95, 7, f"Label Child : {label_child}/16  ({'OK' if lc_ok else 'NOK'})", border=1, fill=True, align="C")
    pdf.ln()

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# HELPER : redimensionner image pour affichage
# ─────────────────────────────────────────────
def resize_for_display(img_rgb: np.ndarray, max_width: int = 700) -> np.ndarray:
    """Réduit l'image si elle dépasse max_width pixels de large."""
    h, w = img_rgb.shape[:2]
    if w <= max_width:
        return img_rgb
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────
# ANALYSE PRINCIPALE
# ─────────────────────────────────────────────
def run_analysis(img: np.ndarray, detect_model, classify_model):
    results = detect_model.predict(img, conf=0.5)
    boxes   = results[0].boxes

    img_draw        = img.copy()
    detected_counts = {}
    crops_display   = []

    COLOR_MOTHER = (0, 165, 255)
    COLOR_CHILD  = (180, 0, 180)
    COLOR_OTHER  = (255, 255, 0)

    # ── Épaisseur des bounding boxes ──
    # Adaptée à la résolution de l'image : ~0.4% de la diagonale, min 3 px
    diag   = int((img.shape[0]**2 + img.shape[1]**2) ** 0.5)
    BOX_THICKNESS = max(3, diag // 250)

    for box in boxes:
        cls   = int(box.cls[0])
        label = results[0].names[cls]
        detected_counts[label] = detected_counts.get(label, 0) + 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])

        if label.lower() in ["capacitor", "capacitor2"]:
            crop = img[y1:y2, x1:x2]
            if crop.size != 0:
                polarity, pol_conf = predict_inversion(classify_model, crop)
                color = (0, 255, 0) if polarity == "OK" else (0, 0, 255)
                text  = f"{label} - {polarity} ({pol_conf:.2f})"
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                crops_display.append((crop, text))
        elif label == "Label Mother":
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), COLOR_MOTHER, BOX_THICKNESS + 1)
        elif label == "Label Child":
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), COLOR_CHILD, BOX_THICKNESS)
        else:
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), COLOR_OTHER, BOX_THICKNESS)

    # ── Affichage image annotée (taille réduite) ──
    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    img_display  = resize_for_display(img_draw_rgb, max_width=700)

    col_img, col_legend = st.columns([3, 1])
    with col_img:
        st.image(img_display,
                 caption="Résultat : Détection + Analyse Polarité",
                 use_container_width=False)
    with col_legend:
        st.markdown("**Légende**")
        st.markdown("🟠 Label Mother")
        st.markdown("🟣 Label Child")
        st.markdown("🟡 Autres composants")
        st.markdown("🟢 Capacitor OK")
        st.markdown("🔴 Capacitor Inversé")

    if crops_display:
        st.subheader("🔍 Détail des condensateurs")
        cols = st.columns(3)
        for i, (crop, text) in enumerate(crops_display):
            with cols[i % 3]:
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=text)

    # ── Tableau d'analyse ──
    st.subheader("📊 Tableau d'analyse")
    table_data     = []
    inverted_found = any("Inverted" in t for _, t in crops_display)

    for comp, expected in golden_board.items():
        detected = detected_counts.get(comp, 0)
        if detected == expected:
            decision = "✅ OK"
        elif detected < expected:
            decision = f"❌ {expected - detected} manquant(s)"
        else:
            decision = f"⚠️ {detected - expected} en surplus"
        table_data.append([comp, expected, detected, decision])

    st.dataframe(pd.DataFrame(table_data,
        columns=["Composant", "Quantité attendue", "Quantité détectée", "Décision"]),
        use_container_width=True)

    # ── Labels ──
    st.subheader("🏷️ Résumé des étiquettes")
    n_mother = detected_counts.get("Label Mother", 0)
    n_child  = detected_counts.get("Label Child",  0)
    c1, c2   = st.columns(2)
    with c1:
        (st.success if n_mother == 1 else st.error if n_mother == 0 else st.warning)(
            f"{'✅' if n_mother==1 else '❌' if n_mother==0 else '⚠️'} Label Mother : {n_mother}/1"
            + ("" if n_mother==1 else " — ABSENTE" if n_mother==0 else " — SURPLUS"))
    with c2:
        (st.success if n_child==16 else st.error if n_child<16 else st.warning)(
            f"{'✅' if n_child==16 else '❌' if n_child<16 else '⚠️'} Label Child : {n_child}/16"
            + ("" if n_child==16 else f" — {16-n_child} manquante(s)" if n_child<16 else f" — {n_child-16} en surplus"))

    # ── Décision finale ──
    st.subheader("🏁 Décision Finale")
    reasons = []
    if any("manquant" in r[-1] for r in table_data): reasons.append("composant(s) manquant(s)")
    if any("surplus"  in r[-1] for r in table_data): reasons.append("composant(s) en surplus")
    if inverted_found:  reasons.append("composant(s) inversé(s)")
    if n_mother != 1:   reasons.append(f"Label Mother incorrecte ({n_mother}/1)")
    if n_child  != 16:  reasons.append(f"Label Child incorrecte ({n_child}/16)")

    if reasons:
        verdict = f"PCB DÉFECTUEUX — {', '.join(reasons)}"
        st.error(f"❌ {verdict}")
    else:
        verdict = "PCB CONFORME"
        st.success("✅ PCB CONFORME")

    # ════════════════════════════════════════════
    # SECTION EXPORT
    # ════════════════════════════════════════════
    st.markdown("---")
    st.subheader("💾 Exporter le rapport")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    col_csv, col_pdf = st.columns(2)

    # ── Export CSV ──
    with col_csv:
        csv_bytes = build_csv(table_data, n_mother, n_child, verdict, timestamp)
        st.download_button(
            label="📄 Télécharger CSV",
            data=csv_bytes,
            file_name=f"inspection_pcb_{ts_file}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Ouvrir dans Excel (séparateur : point-virgule)",
        )



# ═══════════════════════════════════════════════
# UI — TITRE + VÉRIFICATION MODÈLES
# ═══════════════════════════════════════════════
st.title("🔬 PCB AUTOMATED INSPECTION")
st.markdown("---")

models_ok = True
if DETECT_WEIGHTS is None:
    st.error(f"❌ Aucun fichier `.pt` trouvé dans : `{WEIGHTS_DIR}`")
    models_ok = False
if not CLASSIFY_WEIGHTS.exists():
    st.error(f"❌ Classify introuvable : `{CLASSIFY_WEIGHTS}`")
    models_ok = False
if models_ok:
    with st.expander("ℹ️ Modèles chargés", expanded=False):
        st.success(f"✅ Detect   : `{DETECT_WEIGHTS}`")
        st.success(f"✅ Classify : `{CLASSIFY_WEIGHTS}`")

# ═══════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════
tab_upload, tab_webcam = st.tabs(["📁  Upload Image", "📷  Webcam USB"])

# ──────────────────────────────────────────────
# ONGLET 1 — UPLOAD
# ──────────────────────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader("Sélectionner une image PCB",
                                     type=["jpg","jpeg","png"], key="uploader")
    if uploaded_file and models_ok:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("❌ Impossible de décoder l'image.")
        else:
            st.image(
                resize_for_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_width=700),
                caption="Image chargée",
                use_container_width=False
            )
            with st.spinner("⏳ Chargement des modèles..."):
                dm, cm = load_models(str(DETECT_WEIGHTS), str(CLASSIFY_WEIGHTS))
            with st.spinner("🔍 Analyse en cours..."):
                run_analysis(img, dm, cm)

# ──────────────────────────────────────────────
# ONGLET 2 — WEBCAM USB
# ──────────────────────────────────────────────
with tab_webcam:

    cam_index = st.selectbox(
        "Index de la caméra",
        options=[0, 1, 2],
        index=st.session_state.cam_index,
        format_func=lambda x: f"Caméra {x}" + (" (intégrée)" if x==0 else " (USB)"),
    )
    st.session_state.cam_index = cam_index

    st.markdown("---")

    if not st.session_state.show_analysis:

        st.markdown('<span class="cam-live">● Caméra ACTIVE — flux en direct</span>',
                    unsafe_allow_html=True)
        st.info("📌 Positionnez le PCB devant la caméra, puis appuyez sur **📸 Prendre la photo** quand vous êtes prêt.")

        frame_slot  = st.empty()
        capture_btn = st.button("📸  Prendre la photo", key="btn_capture", type="primary")

        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            st.error(f"❌ Impossible d'ouvrir la caméra {cam_index}. Vérifiez le branchement USB.")
            cap.release()
        else:
            for _ in range(2):
                cap.read()

            last_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_slot.warning("⚠️ Flux caméra interrompu.")
                    break

                last_frame = frame

                # Aperçu live — taille réduite aussi
                frame_slot.image(
                    resize_for_display(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_width=700),
                    caption="🎥 Aperçu live — Webcam USB",
                    use_container_width=False
                )

                if capture_btn:
                    cap.release()
                    st.session_state.captured_frame = last_frame
                    st.session_state.show_analysis  = True
                    st.rerun()
                    break

                time.sleep(0.04)

            cap.release()

    else:
        img = st.session_state.captured_frame

        st.success("✅ Photo capturée !")
        st.image(
            resize_for_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_width=700),
            caption="Photo capturée — Webcam USB",
            use_container_width=False
        )

        st.markdown("---")

        if not models_ok:
            st.error("❌ Modèles non disponibles.")
        else:
            with st.spinner("⏳ Chargement des modèles..."):
                dm, cm = load_models(str(DETECT_WEIGHTS), str(CLASSIFY_WEIGHTS))
            with st.spinner("🔍 Analyse en cours..."):
                run_analysis(img, dm, cm)

        st.markdown("---")
        if st.button("🔄 Nouvelle photo", key="btn_retry"):
            st.session_state.captured_frame = None
            st.session_state.show_analysis  = False
            st.rerun()