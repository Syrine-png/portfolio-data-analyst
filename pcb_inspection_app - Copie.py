import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time

# CONFIG PAGE
st.set_page_config(page_title="PCB Inspection", layout="wide")

# PATHS
BASE_DIR = Path("C:/inspection")
WEIGHTS_DIR = BASE_DIR / "runs/detect/pcb_v2-5/weights"

# 🔥 MODÈLE MODIFIÉ
CLASSIFY_WEIGHTS = BASE_DIR / "best_vit_model.pt"

def find_detect_weights(weights_dir):
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    return list(weights_dir.glob("*.pt"))[0]

DETECT_WEIGHTS = find_detect_weights(WEIGHTS_DIR)

# GOLDEN BOARD
golden_board = {
    "relay": 32,
    "MOV": 16,
    "Capacitor": 16,
    "Capacitor2": 32,
    "Label Mother": 1,
    "Label Child": 16,
}

# MODEL LOAD
@st.cache_resource
def load_models():
    import torch, timm
    from ultralytics import YOLO

    detect_model = YOLO(DETECT_WEIGHTS)

    state_dict = torch.load(CLASSIFY_WEIGHTS, map_location="cpu")

    if isinstance(state_dict, dict):
        state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))

    # ✔️ correspond à ton modèle (21MB)
    classify_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)
    classify_model.load_state_dict(state_dict)
    classify_model.eval()

    return detect_model, classify_model

# PREDICTION
def predict_inversion(model, crop):
    import torch
    import torch.nn.functional as F

    crop = cv2.resize(crop, (224, 224))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0
    crop = torch.tensor(crop).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        out = model(crop)
        prob = F.softmax(out, dim=1)[0]

    idx = prob.argmax().item()
    conf = prob[idx].item()

    return ("Inverted", conf) if idx == 0 else ("OK", conf)

# ANALYSIS
def run_analysis(img, detect_model, classify_model):

    results = detect_model.predict(img, conf=0.5)
    boxes = results[0].boxes

    img_draw = img.copy()
    detected_counts = {}

    for box in boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]

        detected_counts[label] = detected_counts.get(label, 0) + 1

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if label.lower() in ["capacitor", "capacitor2"]:
            crop = img[y1:y2, x1:x2]
            if crop.size != 0:
                pol, conf = predict_inversion(classify_model, crop)
                color = (0,255,0) if pol=="OK" else (0,0,255)
                cv2.rectangle(img_draw, (x1,y1),(x2,y2), color, 3)
        else:
            cv2.rectangle(img_draw, (x1,y1),(x2,y2), (255,255,0), 2)

    st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))

    # TABLE
    table = []
    for comp, expected in golden_board.items():
        detected = detected_counts.get(comp,0)
        status = "OK" if detected==expected else "NOK"
        table.append([comp, expected, detected, status])

    st.dataframe(pd.DataFrame(table,
        columns=["Composant","Attendu","Détecté","Status"]))

# UI
st.title("PCB Inspection System")

uploaded = st.file_uploader("Upload PCB image")

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    detect_model, classify_model = load_models()

    run_analysis(img, detect_model, classify_model)