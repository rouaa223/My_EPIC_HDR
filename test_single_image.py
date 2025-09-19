# test_single_image.py

import torch
import math
import numpy as np
import imageio
from compressai.zoo import image_models
from compressai.losses.rate_distortion import RateDistortionLoss
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage.color as color 
from matplotlib import pyplot as plt

# --- Configuration ---
checkpoint_path = "checkpoint_quality1.pth.tar"
original_image_path = "leadenhall_market_4k.hdr"
output_dir = "test_results"
image_name = os.path.splitext(os.path.basename(original_image_path))[0]  
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'hdr'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'ldr'), exist_ok=True)

# Fonction de tonemapping (copiée de train.py)
tonemap = lambda x: (np.log(np.clip(x, 0, 1) * 5000 + 1) / np.log(5000 + 1) * 255).astype(np.uint8)


def _save_image(img, path, name):
    """Sauvegarde une image LDR depuis l'espace HSV."""
    d_max = 300
    d_min = 5
    hdr_h = img.data[0].permute(1, 2, 0)
    t = hdr_h[:, :, 2].cpu()
    t[t > d_max] = d_max
    t[t < d_min] = d_min
    t = (t - d_min) / (d_max - d_min)
    t = (t ** (1 / 2.2))
    hdr_h = hdr_h.cpu().numpy()
    hdr_h[:, :, 2] = t.squeeze().cpu().numpy()
    hdr_h[:, :, 1] = hdr_h[:, :, 1] * 0.6
    result = color.hsv2rgb(hdr_h)
    sz = result.shape
    result1 = np.zeros((sz))
    result1[:, :, 0] = (result[:, :, 0] - np.min(result[:, :, 0])) / (np.max(result[:, :, 0]) - np.min(result[:, :, 0]))
    result1[:, :, 1] = (result[:, :, 1] - np.min(result[:, :, 1])) / (np.max(result[:, :, 1]) - np.min(result[:, :, 1]))
    result1[:, :, 2] = (result[:, :, 2] - np.min(result[:, :, 2])) / (np.max(result[:, :, 2]) - np.min(result[:, :, 2]))
    plt.imsave(path + name[:-4] + '.png', result1)


# --- 1. Charger le checkpoint ---
print("🔍 Chargement du checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
args_lmbda = checkpoint.get('lmbda', 200)
print(f"✅ Checkpoint chargé. Lambda utilisé: {args_lmbda}")

# --- 2. Créer les modèles ---
print("🏗️ Création des modèles...")
net1 = image_models["mbt2018"](quality=1)
net2 = image_models["ldr2hdr"](quality=1)
net = image_models["end2end"](net1, net2)

# --- 3. Charger les poids du modèle ---
try:
    # Essayez d'abord avec ["state_dict"]
    net.load_state_dict(checkpoint["state_dict"])
except KeyError:
    # Si ça échoue, chargez directement le checkpoint
    net.load_state_dict(checkpoint)
net.eval()
device = next(net.parameters()).device
print(f"✅ Modèle 'end2end' prêt sur {device}.")

# --- 4. Charger l'image HDR originale ---
print(f"🖼️ Chargement de l'image : {original_image_path}")
reference = imageio.imread(original_image_path).astype(np.float32)
img_tensor = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to(device)

# Calculer s_max
s_max = torch.tensor([reference.max()]).to(device).to(torch.float32)

# Sauvegarder l'image HDR d'origine
imageio.imwrite(os.path.join(output_dir, f"{image_name}.hdr"), reference)
print(f"✅ Image HDR originale sauvegardée.")

# --- 5. Inférence ---
print("🚀 Inférence en cours...")
with torch.no_grad():
    out_net1, out_net2 = net(img_tensor, s_max, img_tensor)

# --- 6. Extraire les pertes (pour le bit-rate) ---
N, _, H, W = img_tensor.size()
num_pixels = N * H * W

bpp_loss1 = sum(
    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
    for likelihoods in out_net1["likelihoods"].values()
).item()

bpp_loss2 = sum(
    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
    for likelihoods in out_net2["likelihoods"].values()
).item()

total_bpp = bpp_loss1 + bpp_loss2

# --- 7. Sauvegarder l'image HDR reconstruite ---
hdr_recon = out_net2["hdr"]
hdr_recon_max = torch.max(torch.max(torch.max(hdr_recon, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
hdr_recon_normalized = hdr_recon / (hdr_recon_max + 1e-30)
hdr_rgb = hdr_recon_normalized.squeeze().permute(1, 2, 0).cpu().numpy()

reconstructed_hdr_path = os.path.join(output_dir, 'hdr', f"{image_name}_hdr.hdr")
imageio.imwrite(reconstructed_hdr_path, hdr_rgb)
print(f"✅ Image HDR reconstruite (.hdr) sauvegardée.")

# Version tonemappée pour affichage
rgb8_h_tm = tonemap(hdr_rgb / np.max(hdr_rgb))
tm_path = os.path.join(output_dir, 'hdr', f"{image_name}_h_tm.png")
imageio.imwrite(tm_path, rgb8_h_tm)
print(f"✅ Image HDR tonemappée (.png) sauvegardée.")

# --- 8. Sauvegarder l'image LDR reconstruite ---
ldr_hat = out_net1["ldr_x_hat"]
ldr_hat_v = ldr_hat[:, 2, :, :].unsqueeze(1)
ldr_hat_hs = ldr_hat[:, 0:2, :, :]
ldr_out_v2 = (300 - 5) * ldr_hat_v + 5
ldr_out2 = torch.cat([ldr_hat_hs, ldr_out_v2], dim=1)

_save_image(ldr_out2, os.path.join(output_dir, 'ldr'), f"{image_name}.png")
print(f"✅ Image LDR reconstruite (.png) sauvegardée.")

# --- 9. Calculer PSNR-HDR ---
def calculate_psnr_hdr(ref, rec):
    ref_log = np.log(ref + 1e-6)
    rec_log = np.log(rec + 1e-6)
    return psnr(ref_log, rec_log, data_range=ref_log.max() - ref_log.min())

psnr_value = calculate_psnr_hdr(reference, hdr_rgb)
print(f"📊 PSNR-HDR: {psnr_value:.4f} dB")

# --- 10. Afficher le résultat final ---
print("\n" + "="*50)
print("RÉSULTAT FINAL")
print("="*50)
print(f"Image testée        : {image_name}.hdr")
print(f"Checkpoint utilisé  : {checkpoint_path}")
print(f"PSNR-HDR            : {psnr_value:.4f} dB")
print(f"Bpp Loss1 (HDR→LDR) : {bpp_loss1:.4f} bpp")
print(f"Bpp Loss2 (LDR→HDR) : {bpp_loss2:.4f} bpp")
print(f"Bit-Rate Total      : {total_bpp:.4f} bpp")
print("="*50)