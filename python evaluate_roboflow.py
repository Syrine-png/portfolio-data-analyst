from roboflow import Roboflow
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==========================
# CONFIGURATION
# ==========================

API_KEY = "A5CKGBfCx8oSKTGelTPc"

rf = Roboflow(api_key=API_KEY)

project = rf.workspace("syrines-workspace").project("inversion_detection")
model = project.version(2).model

TEST_DIR = r"C:\Users\syrine mlika\Downloads\inversion_detection.v1i.folder\test"

OUTPUT_DIR = Path(r"C:\inspection\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# DONNÉES
# ==========================

y_true = []
y_score = []

classes = sorted([
    d.name for d in Path(TEST_DIR).iterdir()
    if d.is_dir()
])

print("Classes détectées :", classes)

# ==========================
# LOOP
# ==========================

for class_name in classes:

    class_dir = Path(TEST_DIR) / class_name
    images = list(class_dir.glob("*.*"))

    for img_path in tqdm(images, desc=class_name):

        try:
            pred = model.predict(str(img_path)).json()
            result = pred["predictions"][0]

            top_class = result["top"]
            confidence = result["confidence"]

            if top_class == "OK":
                score = confidence
            else:
                score = 1 - confidence

            y_score.append(score)
            y_true.append(1 if class_name == "OK" else 0)

        except Exception as e:
            print(f"Erreur sur {img_path}")
            print(e)

# ==========================
# NUMPY
# ==========================

y_true = np.array(y_true)
y_score = np.array(y_score)

# ==========================
# CONFUSION MATRIX
# ==========================

best_threshold = 0.5
y_pred = (y_score >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)

print("\nMatrice de confusion :")
print(cm)

# ==========================
# PLOT + SAVE IMAGE
# ==========================

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

tick_labels = ["Inverted", "OK"]
plt.xticks([0, 1], tick_labels)
plt.yticks([0, 1], tick_labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

print("\nConfusion matrix saved in:", OUTPUT_DIR)