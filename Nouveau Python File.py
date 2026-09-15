import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────
# PATHS
# ─────────────────────────────
model_path = "C:/inspection/best_vit_model.pt"
test_path = "C:/inspection/inversion_detection.v1i.folder/test"

# ─────────────────────────────
# TRANSFORM
# ─────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────
# DATASET
# ─────────────────────────────
test_dataset = datasets.ImageFolder(test_path, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes
print("Classes :", class_names)

# ─────────────────────────────
# MODEL
# ─────────────────────────────
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2)

state_dict = torch.load(model_path, map_location="cpu")

if isinstance(state_dict, dict):
    state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))

model.load_state_dict(state_dict)
model.eval()

# ─────────────────────────────
# TEST
# ─────────────────────────────
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# ─────────────────────────────
# METRICS
# ─────────────────────────────
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall    = recall_score(y_true, y_pred, average="weighted")
f1        = f1_score(y_true, y_pred, average="weighted")

print("\n🎯 PERFORMANCE GLOBALE")
print(f"Accuracy  : {accuracy*100:.4f} %")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ─────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────
cm = confusion_matrix(y_true, y_pred)

print("\n📊 MATRICE DE CONFUSION :\n")
print(cm)

# ─────────────────────────────
# RAPPORT COMPLET
# ─────────────────────────────
print("\n📊 CLASSIFICATION REPORT :\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ─────────────────────────────
# PLOT MATRICE
# ─────────────────────────────
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()

plt.title("Confusion Matrix - ViT Model")
plt.show()