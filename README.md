# Learned HDR Image Compression for Perceptually Optimal Storage and Display

## Introduction
This repository contains the official pytorch implementation of the paper ["Learned HDR Image Compression for Perceptually Optimal Storage and Display"] by Peibei Cao, Haoyu Chen, Jingzhe Ma, Yu-Chieh Yuan, Zhiyong Xie, Xin Xie, Haiqing Bai, and Kede Ma, The European Conference on Computer Vision, 2024.
# Projet de Compression HDR avec Apprentissage Profond

Ce projet implémente un système de compression d'images HDR (High Dynamic Range) basé sur l'apprentissage profond, inspiré du papier [EPIC-HDR: Perceptually Informed Compression for High Dynamic Range Images](https://arxiv.org/html/2407.13179v1).

## Architecture

Le modèle utilise une architecture en deux étapes :
1. **`mbt2018`** : Compresse l'image HDR en une image LDR (Low Dynamic Range) + métadonnées.
2. **`ldr2hdr`** : Reconstruit l'image HDR à partir de l'image LDR compressée.

Les deux modèles sont combinés dans un pipeline **end-to-end** (`end2end`) pour un entraînement conjoint.

## Structure du projet
EPIC-HDR/
├── train.py                # Script principal d'entraînement
├── test_single_image.py    # Script de test sur une seule image
├── setup_dataset.py        # Script pour diviser le dataset en train/test
├── my_dataset/             # Dossier contenant les images HDR (.hdr, .exr)
│   ├── train.txt           # Liste des images d'entraînement
│   ├── test.txt            # Liste des images de test
│   ├── train/              # Images d'entraînement
│   └── test/               # Images de test
├── output/                 # Résultats de la reconstruction
├── venv/                   # Environnement virtuel Python
└── checkpoint/            # Modèles sauvegardés vous trouvez checkpoint_quality1,2,3,5,7(paramètres avec training sur un dataset contenat environ 2000 images) 
                             et checkpoint.pth (paramètre du modèle entrainé avec un dataset contenat environ 7000 images)



## Prérequis

- Python 3.9
- PyTorch 2.7.1
- NumPy 1.26.4
- cv2 4.8.1

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux

# Installer les dépendances
pip install requirements.txt (en cas de problème de version uninstall les bibliothèques qui cause des problèmes et les installer manuellement)
#voir les config possible 
python train.py --help
#entrainement du modèle
python train.py \
  --dataset my_dataset \
  --test_dataset my_dataset \
  --results_savepath output \
  --epochs 50 \
  --batch-size 1 \
  --patch-size 512 512 \
  --cuda \
  --save \
  --lambda 200 \
  --qualities "1,2,3,4,5,6,7,8"  #vous pouvez modifier la liste (choisir un ou plusieurs)

#tester une image 
python test_single_image.py #vous devez modifier le script avant de tester :
# Ces lignes de code : 
  # --- Configuration ---
checkpoint_path = "checkpoint_quality1.pth.tar"
original_image_path = "leadenhall_market_4k.hdr"
...
  # --- 2. Créer les modèles ---
print("🏗️ Création des modèles...")
net1 = image_models["mbt2018"](quality=1)#modifier la qualité aussi du modèle
net2 = image_models["ldr2hdr"](quality=1)
net = image_models["end2end"](net1, net2)


