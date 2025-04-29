import os
import numpy as np
import torch
import tifffile
from tqdm import tqdm

# === Configuration ===
data_dir = "../data/newData"  # ← change this
channels = ["nadh", "fad", "shg"]
percentile = 99.9

# === Step 1: Collect all image paths and pixel values ===
pixel_values = {ch: [] for ch in channels}
image_paths = {ch: [] for ch in channels}

print("Collecting pixel values for percentile normalization...")

for root, _, files in os.walk(data_dir):
    for fname in files:
        for ch in channels:
            if fname.lower() == f"{ch}.tiff":
                full_path = os.path.join(root, fname)
                img = tifffile.imread(full_path).astype(np.float32)
                img = np.nan_to_num(img, nan=1.0, posinf=1.0, neginf=0.0)

                pixel_values[ch].append(img.flatten())
                image_paths[ch].append(full_path)

# === Step 2: Compute 99.5th percentile per channel ===
channel_percentiles = {}

for ch in channels:
    if pixel_values[ch]:
        all_pixels = np.concatenate(pixel_values[ch])
        p = np.percentile(all_pixels, percentile)
        channel_percentiles[ch] = p
        print(f"{ch.upper()} 100th percentile (max): {p:.4f}")
    else:
        print(f"Warning: No {ch}.tiff images found.")
        channel_percentiles[ch] = None

# === Step 3: Normalize and save ===
print("\nNormalizing and saving images...")

for ch in channels:
    pval = channel_percentiles[ch]
    if pval is None or pval == 0:
        continue

    for path in tqdm(image_paths[ch], desc=f"Normalizing {ch.upper()}"):
        img = tifffile.imread(path).astype(np.float32)
        img = np.nan_to_num(img, nan=1.0, posinf=1.0, neginf=0.0)

        norm_img = img / pval
        norm_img = np.clip(norm_img, 0, 1)

        out_path = os.path.join(os.path.dirname(path), f"{ch}_norm.tiff")
        tifffile.imwrite(out_path, norm_img.astype(np.float32))

print("\n✅ Done. Normalized images saved with '_norm' suffix.")
