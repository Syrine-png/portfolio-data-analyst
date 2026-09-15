import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = r"C:\inspection\best_vit_model.pt"
test_path  = r"C:\inspection\inversion_detection.v1i.folder\test"

# =========================
# MODEL (ATTENTION : Tiny)
# =========================
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path))
model.to(device)

# =========================
# DATASET
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# TEST
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

acc = 100 * correct / total
print("🎯 TEST ACCURACY (inspection model):", acc)