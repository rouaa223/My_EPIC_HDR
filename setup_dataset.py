# setup_dataset.py

import os
import shutil
import random
from pathlib import Path

# Paramètres
root_dir = Path("my_dataset")
train_dir = root_dir / "train"
test_dir = root_dir / "test"
train_ratio = 0.8  # 80% train, 20% test

# Créer les dossiers train et test
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Vérifier si train et test sont vides
if not any(train_dir.iterdir()) and not any(test_dir.iterdir()):
    print(" Répartition des images en train/test...")

    # Trouver toutes les images .hdr (minuscules et majuscules)
    all_images = list(root_dir.glob("*.hdr")) + list(root_dir.glob("*.HDR"))
    
    if not all_images:
        raise FileNotFoundError(f"Aucune image .hdr trouvée dans {root_dir}")

    # Mélanger pour un échantillonnage aléatoire
    random.shuffle(all_images)

    # Diviser
    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    # Copier les fichiers
    for src in train_files:
        dst = train_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)

    for src in test_files:
        dst = test_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)

    print(f"{len(train_files)} images copiées dans {train_dir}")
    print(f"{len(test_files)} images copiées dans {test_dir}")

    # Créer train.txt (uniquement le nom du fichier)
    with open(root_dir / "train.txt", "w") as f:
        for img in train_files:
            f.write(f"{img.name}\n")

    # Créer test.txt
    with open(root_dir / "test.txt", "w") as f:
        for img in test_files:
            f.write(f"{img.name}\n")

    print("Fichiers train.txt et test.txt créés.")
else:
    print(" Les dossiers train/ ou test/ ne sont pas vides — pas de répartition.")
    print("   Supprimez-les ou videz-les si vous voulez refaire la répartition.")
