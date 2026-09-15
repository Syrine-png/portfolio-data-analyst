"""
=============================================================
  Entraînement ViT — Classification Condensateurs OK / Inverted
  Vision Transformer (google/vit-base-patch16-224) via HuggingFace
  Dataset exporté depuis Roboflow (format Classification)
  Auteur : Script PFE — Zollner Elektronik Tunisie / ENISo
=============================================================

Structure attendue du dataset Roboflow (format Classification) :
  dataset/
    train/
      Inverted/    ← condensateurs inversés
      ok/          ← condensateurs correctement insérés
    valid/
      Inverted/
      ok/
    test/          (optionnel)
      Inverted/
      ok/

Installation des dépendances :
    pip install torch torchvision transformers timm scikit-learn
    pip install matplotlib seaborn tqdm

Lancer avec :
    python train_vit_capacitor.py --data_dir ./dataset --epochs 30

Le modèle sauvegardé (best_vit_model.pt) est directement
compatible avec l'application Streamlit (pcb_lean_ai.py).
=============================================================
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ─────────────────────────────────────────
#  Paramètres par défaut
# ─────────────────────────────────────────
DEFAULTS = dict(
    data_dir    = r"C:\inspection\inversion_detection.v1i.folder",
    epochs      = 10,        # Classification binaire + ViT pré-entraîné → converge en 5-8 époques
    batch_size  = 32,        # ViT est plus lourd — batch réduit vs EfficientNet
    lr          = 2e-5,      # Learning rate faible pour fine-tuning ViT
    img_size    = 224,       # ViT-base-patch16-224 attend exactement 224×224
    output_dir  = r"C:\inspection\results_vit",
    patience    = 4,         # Arrêt rapide si stagnation (early stopping agressif)
    seed        = 42,
    freeze_layers = 10,      # Geler 10/12 blocs — seuls les 2 derniers + tête s'adaptent
    warmup_epochs = 2,       # Warmup court cohérent avec 10 époques max
)

# Ordre alphabétique ASCII = ordre PyTorch ImageFolder
# ord('I') = 73  <  ord('O') = 79
# → "Inverted" vient AVANT "OK", même si Roboflow affiche OK en premier
# → index 0 = Inverted, index 1 = OK  — NE PAS INVERSER
CLASS_NAMES = ["Inverted", "OK"]


# ══════════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════════
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"✅  GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("⚠️  CPU uniquement — l'entraînement sera plus lent avec ViT.")
    return dev


# ══════════════════════════════════════════════════════════════════
#  MODÈLE ViT
# ══════════════════════════════════════════════════════════════════
def build_vit_model(num_classes: int, freeze_layers: int, device: torch.device):
    """
    Charge ViT-Base/16 pré-entraîné sur ImageNet-21k (via timm)
    et adapte la tête de classification.

    Architecture :
      - Patch embedding  : 16×16 patches → séquence de tokens
      - Transformer      : 12 blocs d'attention multi-têtes
      - [CLS] token      → tête linéaire → num_classes

    Stratégie de fine-tuning :
      - Les `freeze_layers` premiers blocs sont gelés (feature extraction)
      - Les blocs suivants + la tête sont entraînés (adaptation au domaine)
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "Le package 'timm' est requis.\n"
            "Installez-le avec : pip install timm"
        )

    # Chargement ViT-Base/16 pré-entraîné (ImageNet-21k → ImageNet-1k)
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )

    # ── Gel des premières couches (feature extraction basse résolution) ──
    # Couches à geler : patch_embed + positional embedding + blocs 0..freeze_layers-1
    for param in model.patch_embed.parameters():
        param.requires_grad = False

    for i, block in enumerate(model.blocks):
        if i < freeze_layers:
            for param in block.parameters():
                param.requires_grad = False

    # La tête de classification est toujours entraînable
    for param in model.head.parameters():
        param.requires_grad = True

    # Résumé des paramètres
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠  Modèle : ViT-Base/16-224")
    print(f"   Paramètres totaux    : {total/1e6:.1f}M")
    print(f"   Paramètres entraînés : {trainable/1e6:.1f}M ({100*trainable/total:.1f}%)")
    print(f"   Blocs gelés          : {freeze_layers}/12\n")

    return model.to(device)


# ══════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════
def build_loaders(data_dir, img_size, batch_size):
    """
    Transformations adaptées au ViT :
    - Normalisation ImageNet (même que le pré-entraînement)
    - Augmentations modérées pour ne pas dégrader les features pré-entraînées
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomRotation(10),
        # RandAugment améliore la généralisation pour ViT
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Erasing aléatoire : simule des occlusions partielles
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "valid", transform=val_tf)

    test_path = data_dir / "test"
    test_ds   = datasets.ImageFolder(test_path, transform=val_tf) if test_path.exists() else None

    # Vérification de l'ordre des classes (CRITIQUE pour la cohérence Streamlit)
    print(f"\n📂  Classes détectées (ordre PyTorch) : {train_ds.classes}")
    print(f"    ⚠️  Vérification : index 0 = '{train_ds.classes[0]}' | index 1 = '{train_ds.classes[1]}'")
    assert train_ds.classes[0] == "Inverted" and train_ds.classes[1] == "OK", (
        f"\n❌  ERREUR : ordre des classes inattendu → {train_ds.classes}\n"
        f"    Vérifiez que vos dossiers s'appellent exactement 'Inverted' et 'OK'."
    )
    print(f"    ✅  Ordre correct : Inverted=0, OK=1")
    print(f"\n    Train : {len(train_ds)} images | Val : {len(val_ds)} images", end="")
    if test_ds:
        print(f" | Test : {len(test_ds)} images")
    else:
        print()

    from collections import Counter
    c = Counter(train_ds.targets)
    print(f"    Distribution train → " +
          " | ".join(f"{train_ds.classes[k]}: {v}" for k, v in sorted(c.items())))

    # num_workers=0 sur Windows pour éviter les problèmes de multiprocessing
    nw = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = (DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                               num_workers=nw, pin_memory=True) if test_ds else None)

    return train_loader, val_loader, test_loader, train_ds.classes


# ══════════════════════════════════════════════════════════════════
#  BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision (AMP) pour accélérer sur GPU
            with torch.amp.autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False, ncols=80):
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        probs = torch.softmax(out, dim=1)

        running_loss += loss.item() * imgs.size(0)
        preds         = out.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ══════════════════════════════════════════════════════════════════
#  SCHEDULER AVEC WARMUP
# ══════════════════════════════════════════════════════════════════
def build_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Warmup linéaire pendant `warmup_epochs`,
    puis décroissance cosinus jusqu'à la fin.
    Stratégie recommandée pour le fine-tuning de ViT.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════
def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'Entraînement — ViT Classification Condensateurs",
                 fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    best_e = int(np.argmin(history["val_loss"])) + 1
    axes[0].axvline(best_e, color="green", linestyle="--", alpha=0.7,
                    label=f"Meilleur (ép.{best_e})")
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss (Cross-Entropy)")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [v*100 for v in history["train_acc"]], "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, [v*100 for v in history["val_acc"]],   "r-o", markersize=4, label="Val Acc")
    best_e2 = int(np.argmax(history["val_acc"])) + 1
    axes[1].axvline(best_e2, color="green", linestyle="--", alpha=0.7,
                    label=f"Meilleur (ép.{best_e2})")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / "01_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Courbes d'entraînement → {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir, split="Val"):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Matrice de Confusion — {split}", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Comptes bruts", "Normalisée (%)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
        ax.set_title(title)

    plt.tight_layout()
    path = Path(output_dir) / f"02_confusion_matrix_{split.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Matrice de confusion ({split}) → {path}")


def plot_roc_pr_curves(y_true, y_probs, class_names, output_dir, split="Val"):
    # Classe positive = Inverted (index 0)
    pos_idx  = 0
    y_scores = y_probs[:, pos_idx]

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_idx)
    roc_auc      = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_idx)
    ap = average_precision_score((y_true == pos_idx).astype(int), y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Courbes ROC et PR — {split} (classe positive : Inverted)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire")
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("Courbe ROC"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall, precision, "r-", lw=2, label=f"AP = {ap:.4f}")
    axes[1].fill_between(recall, precision, alpha=0.1, color="red")
    axes[1].set_xlabel("Rappel"); axes[1].set_ylabel("Précision")
    axes[1].set_title("Courbe Précision-Rappel"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / f"03_roc_pr_curves_{split.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Courbes ROC & PR ({split}) → {path}")
    return roc_auc, ap


def plot_metrics_summary(report_dict, roc_auc, ap, class_names, output_dir, split="Val"):
    rows = []
    for cn in class_names:
        r = report_dict.get(cn, {})
        rows.append([cn,
                     f"{r.get('precision', 0):.4f}",
                     f"{r.get('recall',    0):.4f}",
                     f"{r.get('f1-score',  0):.4f}",
                     str(r.get('support', '—'))])
    rows.append(["", "", "", "", ""])
    rows.append(["accuracy",   "—", "—", f"{report_dict['accuracy']:.4f}", "—"])
    rows.append(["macro avg",
                 f"{report_dict['macro avg']['precision']:.4f}",
                 f"{report_dict['macro avg']['recall']:.4f}",
                 f"{report_dict['macro avg']['f1-score']:.4f}", "—"])
    rows.append(["ROC-AUC",       "—", "—", f"{roc_auc:.4f}", "—"])
    rows.append(["Avg Precision", "—", "—", f"{ap:.4f}",      "—"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    cols = ["Classe", "Précision", "Rappel", "F1-Score", "Support"]
    tbl  = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and rows[r-1][0] in class_names:
            cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "#ffffff")
        elif r > 0 and rows[r-1][0] in ("accuracy", "macro avg", "ROC-AUC", "Avg Precision"):
            cell.set_facecolor("#d5e8d4")
    ax.set_title(f"Récapitulatif des Métriques — {split}",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = Path(output_dir) / f"04_metrics_summary_{split.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Tableau métriques ({split}) → {path}")


def plot_lr_schedule(lrs, output_dir):
    if not lrs:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(lrs, "g-", lw=2)
    plt.xlabel("Époque"); plt.ylabel("Learning Rate")
    plt.title("Évolution du Learning Rate (Warmup + Cosine Decay)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    path = Path(output_dir) / "05_lr_schedule.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  LR schedule → {path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main(args):
    set_seed(args.seed)
    device     = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data
    train_loader, val_loader, test_loader, class_names = build_loaders(
        args.data_dir, args.img_size, args.batch_size
    )
    num_classes = len(class_names)

    # ── Modèle ViT
    model = build_vit_model(num_classes, args.freeze_layers, device)

    # ── Loss + Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # ── AMP (mixed precision) si GPU disponible
    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("⚡  Mixed Precision (AMP) activé — entraînement accéléré\n")

    # ── Historique
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    lrs     = []
    best_val_loss  = float("inf")
    patience_counter = 0
    best_model_path  = output_dir / "best_vit_model.pt"

    print("─" * 68)
    print(f"{'Ép':>4}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>10}")
    print("─" * 68)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0] * args.lr   # lr absolu

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        lrs.append(cur_lr)

        marker = " ✅" if vl_loss < best_val_loss else ""
        print(f"{epoch:>4}  {tr_loss:>10.4f}  {tr_acc*100:>8.2f}%  "
              f"{vl_loss:>9.4f}  {vl_acc*100:>7.2f}%  {cur_lr:>10.2e}{marker}")

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹  Early stopping à l'époque {epoch} (patience={args.patience})")
                break

    elapsed = time.time() - t0
    print("─" * 68)
    print(f"⏱  Durée totale : {elapsed/60:.1f} min")
    print(f"✅  Meilleur modèle sauvegardé → {best_model_path}\n")

    # ── Chargement du meilleur modèle
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ── Évaluation finale (Validation)
    print("📈  Génération des courbes et métriques…\n")
    _, _, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
    report = classification_report(val_labels, val_preds,
                                   target_names=class_names, output_dict=True)
    print("Rapport de classification (Validation) :")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # ── Graphiques
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, output_dir, split="Val")
    roc_auc, ap = plot_roc_pr_curves(val_labels, val_probs, class_names, output_dir, split="Val")
    plot_metrics_summary(report, roc_auc, ap, class_names, output_dir, split="Val")
    plot_lr_schedule(lrs, output_dir)

    # ── Test set (si disponible)
    if test_loader:
        print("\n🔍  Évaluation sur le Test set…")
        _, _, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, criterion, device
        )
        report_test = classification_report(test_labels, test_preds,
                                            target_names=class_names, output_dict=True)
        print(classification_report(test_labels, test_preds, target_names=class_names))
        plot_confusion_matrix(test_labels, test_preds, class_names, output_dir, split="Test")
        roc_auc_t, ap_t = plot_roc_pr_curves(
            test_labels, test_probs, class_names, output_dir, split="Test"
        )
        plot_metrics_summary(report_test, roc_auc_t, ap_t, class_names, output_dir, split="Test")

    # ── Résumé JSON
    summary = {
        "model"            : "ViT-Base/Patch16-224",
        "freeze_layers"    : args.freeze_layers,
        "best_epoch"       : int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss"    : float(best_val_loss),
        "best_val_acc"     : float(max(history["val_acc"])),
        "val_roc_auc"      : float(roc_auc),
        "val_avg_precision": float(ap),
        "class_names"      : class_names,   # sauvegardé pour référence Streamlit
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  Terminé ! Tous les résultats dans : {output_dir}/")
    print(f"   Meilleure Validation Accuracy : {max(history['val_acc'])*100:.2f}%")
    print(f"   ROC-AUC : {roc_auc:.4f}")
    print(f"\n📌  Pour utiliser dans Streamlit :")
    print(f"   CLASSIFY_WEIGHTS = r'{best_model_path}'")
    print(f"   → et utiliser load_vit_model() dans pcb_lean_ai.py")


# ══════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Entraînement ViT-Base/16 — Classification condensateurs OK/Inverted"
    )
    p.add_argument("--data_dir",      default=DEFAULTS["data_dir"])
    p.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch_size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--img_size",      type=int,   default=DEFAULTS["img_size"])
    p.add_argument("--output_dir",    default=DEFAULTS["output_dir"])
    p.add_argument("--patience",      type=int,   default=DEFAULTS["patience"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--freeze_layers", type=int,   default=DEFAULTS["freeze_layers"],
                   help="Nombre de blocs Transformer à geler (0-12, défaut=8)")
    p.add_argument("--warmup_epochs", type=int,   default=DEFAULTS["warmup_epochs"],
                   help="Nombre d'époques de warmup du learning rate")
    args = p.parse_args()
    main(args)