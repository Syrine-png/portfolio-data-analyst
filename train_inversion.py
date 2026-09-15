import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# 🔧 CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

train_path = r"C:\inspection\inversion_detection.v1i.folder\train"
val_path   = r"C:\inspection\inversion_detection.v1i.folder\valid"

EPOCHS = 30
BATCH_SIZE = 32

# =========================
# 🧠 MODEL
# =========================
model = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=True,
    num_classes=2
)

model.to(device)

# =========================
# 📁 DATASET
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_path, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# ⚙️ TRAIN SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0

# 📊 stockage pour courbes
train_losses = []
val_accuracies = []

# =========================
# 🚀 TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 🔍 VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    # 💾 Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_vit_model.pt")

    scheduler.step()

print("✅ Training terminé. Meilleure accuracy :", best_acc)

# =========================
# 📊 PLOT COURBES
# =========================
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.figure()
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()