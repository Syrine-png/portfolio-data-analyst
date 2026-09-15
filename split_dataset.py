import os, shutil, random
from pathlib import Path

# ── Chemins ──────────────────────────────────────────
BASE = Path("C:/inspection/missing_component_detection.yolov8")
TRAIN_IMG   = BASE / "train/images"
TRAIN_LBL   = BASE / "train/labels"
VALID_IMG   = BASE / "valid/images"
VALID_LBL   = BASE / "valid/labels"

SPLIT = 0.2   # 20% des images iront dans valid

# ── Créer les dossiers valid ──────────────────────────
VALID_IMG.mkdir(parents=True, exist_ok=True)
VALID_LBL.mkdir(parents=True, exist_ok=True)

# ── Lister toutes les images ──────────────────────────
images = list(TRAIN_IMG.glob("*.jpg")) + \
         list(TRAIN_IMG.glob("*.jpeg")) + \
         list(TRAIN_IMG.glob("*.png"))

random.seed(42)
random.shuffle(images)

nb_valid = max(1, int(len(images) * SPLIT))
valid_images = images[:nb_valid]

print(f"Total images      : {len(images)}")
print(f"Images -> valid   : {nb_valid}")
print(f"Images -> train   : {len(images) - nb_valid}")
print()

# ── Déplacer images + labels vers valid ──────────────
for img_path in valid_images:
    # déplacer image
    shutil.move(str(img_path), str(VALID_IMG / img_path.name))

    # déplacer le label correspondant (.txt)
    lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(VALID_LBL / lbl_path.name))
    else:
        print(f"  Attention : pas de label pour {img_path.name}")

print("Fait ! Dossier valid/ créé avec succès.")