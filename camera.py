import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Inspection System",
    page_icon="🔬",
    layout="wide"
)

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path("C:/detection_project")
WEIGHTS_DIR = BASE_DIR / "runs/detect/train2/weights"
CLASSIFY_WEIGHTS = BASE_DIR / "runs/classify/best_Copie.pt"

def find_detect_weights(weights_dir: Path) -> Path:
    """Trouve automatiquement le fichier .pt dans le dossier weights."""
    if not weights_dir.exists():
        return None
    # Priorité : best.pt, sinon n'importe quel .pt
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]
    return None

DETECT_WEIGHTS = find_detect_weights(WEIGHTS_DIR)

# ─────────────────────────────────────────────
# GOLDEN BOARD
# ─────────────────────────────────────────────
golden_board = {
    "relay":      32,
    "MOV":        16,
    "Capacitor":  16,
    "Capacitor2": 32,
}

# ─────────────────────────────────────────────
# CLASSIFICATION FUNCTION
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

    CONFIDENCE_THRESHOLD = 0.70
    if conf < CONFIDENCE_THRESHOLD:
        return "OK", conf

    return classes[idx], conf

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models(detect_path: str, classify_path: str):
    import torch
    import timm
    from ultralytics import YOLO

    detect_model = YOLO(detect_path)

    state_dict = torch.load(classify_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    classify_model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=2
    )
    classify_model.load_state_dict(state_dict)
    classify_model = classify_model.eval()

    return detect_model, classify_model
def capture_usb_photo():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("❌ Impossible d'ouvrir la caméra USB.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Impossible de capturer une image.")
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)

    return temp_file.name

# ─────────────────────────────────────────────
# UI — TITRE
# ─────────────────────────────────────────────
st.title("🔬 PCB AUTOMATED INSPECTION")


# ─────────────────────────────────────────────
# VÉRIFICATION DES MODÈLES
# ─────────────────────────────────────────────
models_ok = True

if DETECT_WEIGHTS is None:
    st.error(f"❌ Aucun fichier `.pt` trouvé dans : `{WEIGHTS_DIR}`")
    st.info("Fichiers présents dans le dossier parent :")
    if WEIGHTS_DIR.parent.exists():
        for f in WEIGHTS_DIR.parent.rglob("*.pt"):
            st.code(str(f))
    models_ok = False

if not CLASSIFY_WEIGHTS.exists():
    st.error(f"❌ Fichier classify introuvable : `{CLASSIFY_WEIGHTS}`")
    models_ok = False

if models_ok:
    st.success(f"✅ Detect weights  : `{DETECT_WEIGHTS}`")
    st.success(f"✅ Classify weights : `{CLASSIFY_WEIGHTS}`")

# ─────────────────────────────────────────────
# UPLOAD IMAGE
# ─────────────────────────────────────────────
st.subheader("📷 Acquisition image")

mode = st.radio(
    "Choisir la source d'image",
    ["Webcam USB", "Upload Image"]
)

path = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload PCB Image",
        type=["jpg", "jpeg", "png"]
    )

    
else:
    if st.button("📸 Prendre une photo avec la webcam USB"):
        path = capture_usb_photo()



    img = cv2.imread(path)

    if img is None:
        st.error("❌ Erreur lors de la lecture de l'image.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Image uploadée")

        with st.spinner("Chargement des modèles..."):
            detect_model, classify_model = load_models(
                str(DETECT_WEIGHTS),
                str(CLASSIFY_WEIGHTS)
            )

        with st.spinner("Analyse en cours..."):
            results = detect_model.predict(path, conf=0.5)
            boxes   = results[0].boxes

        img_draw = img.copy()
        detected_counts = {}
        crops_display   = []

        for box in boxes:
            cls   = int(box.cls[0])
            label = results[0].names[cls]
            detected_counts[label] = detected_counts.get(label, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label.lower() in ["capacitor", "capacitor2"]:
                crop = img[y1:y2, x1:x2]

                if crop.size != 0:
                    polarity, conf = predict_inversion(classify_model, crop)

                    color = (0, 255, 0) if polarity == "OK" else (0, 0, 255)
                    text  = f"{label} - {polarity} ({conf:.2f})"

                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_draw, text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    crops_display.append((crop, text))
            else:
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(img_draw, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ───── IMAGE RÉSULTAT ─────
        st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                 caption="Détection + Résultat Inversion")

        # ───── CROPS ─────
        if crops_display:
            st.subheader("🔍 Composants détectés (crops)")
            cols = st.columns(3)
            for i, (crop, text) in enumerate(crops_display):
                with cols[i % 3]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=text)

        # ───── TABLEAU ─────
        st.subheader("📊 Tableau d'analyse des composants")

        table_data = []
        inverted_found = False

        for comp, expected in golden_board.items():
            detected = detected_counts.get(comp, 0)
            if detected == expected:
                decision = "✅ OK"
            else:
                decision = "❌ Composant manquant"
            table_data.append([comp, expected, detected, decision])

        # Vérifier les inversions dans les crops
        for _, text in crops_display:
            if "Inverted" in text:
                inverted_found = True
                break

        df = pd.DataFrame(table_data, columns=[
            "Composant",
            "Quantité attendue",
            "Quantité détectée",
            "Décision"
        ])

        st.dataframe(df, use_container_width=True)

        # ───── DÉCISION FINALE ─────
        st.subheader("🏁 Décision Finale")

        has_missing = any("manquant" in row[-1] for row in table_data)

        if has_missing or inverted_found:
            reasons = []
            if has_missing:
                reasons.append("composant(s) manquant(s)")
            if inverted_found:
                reasons.append("composant(s) inversé(s)")
            st.error(f"❌ DÉCISION FINALE : PCB DÉFECTUEUX — {', '.join(reasons)}")
        else:
            st.success("✅ DÉCISION FINALE : PCB CONFORME")

    # Nettoyage fichier temporaire
    try:
        os.remove(path)
    except Exception:
        pass