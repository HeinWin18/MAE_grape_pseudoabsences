#!/usr/bin/env python
import numpy as np
import rasterio
import glob
import os
import csv
import argparse
from tqdm import tqdm
from tensorflow.keras.models import load_model
import tensorflow as tf
import model
from patchGan import compute_patch_anomaly_scores
import time

total_start = time.time()
patch_times = []

def run(testpath, imgsize, channels, zdims, lrg, lrd, top_k, encoder_path, output_csv):
    g = load_model('/Users/justin/Desktop/Rast-fAnoGAN/saved_model/generator20260816_224127.h5',
                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    d = load_model('/Users/justin/Desktop/Rast-fAnoGAN/saved_model/discriminator20260816_224127.h5',
                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    encoder = load_model(encoder_path)  # ADDED: load trained encoder

    class Args: pass
    args = Args()
    args.imgsize = imgsize; args.channels = channels
    args.zdims = zdims; args.lrg = lrg; args.lrd = lrd

    all_files = sorted(glob.glob(os.path.join(testpath, '*.tif')))
    test_files = all_files  # test on first 50

    print("Computing global normalization stats...")
    all_raw = []
    for fp in test_files:
        with rasterio.open(fp) as src:
            arr = np.moveaxis(src.read().astype(np.float32), 0, -1)
            if arr.shape == (imgsize, imgsize, channels + 1):
                all_raw.append(arr[:, :, :-1])
    all_raw = np.stack(all_raw, axis=0)
    global_mean = all_raw.mean(axis=(0,1,2), keepdims=True)
    global_std = all_raw.std(axis=(0,1,2), keepdims=True) + 1e-7
    del all_raw

    all_candidates = []

    for fp in tqdm(test_files, desc="Scoring patches (f-AnoGAN)"):
        patch_start = time.time()
        with rasterio.open(fp) as src:
            arr = np.moveaxis(src.read().astype(np.float32), 0, -1)
            if arr.shape != (imgsize, imgsize, channels + 1):
                continue

            b1 = arr[:, :, -1]
            img_raw = arr[:, :, :-1]
            img_norm = np.nan_to_num((img_raw - global_mean) / global_std, nan=0.0)
            mask = (~np.isnan(b1)).astype(np.float32)

            # REPLACED: single encoder forward pass instead of 100-iteration loop
            z = encoder.predict(img_norm.reshape(1, imgsize, imgsize, channels), verbose=0)
            sim_img = g.predict(z, verbose=0)

            patch_scores, _ = compute_patch_anomaly_scores(
                img_norm, sim_img.reshape(imgsize, imgsize, channels),
                d, alpha=0.9, beta=0.1, mask=mask)

            for r in range(imgsize):
                for c in range(imgsize):
                    if mask[r, c] == 0:
                        continue
                    all_candidates.append((patch_scores[r, c], img_raw[r, c, :].tolist()))

        patch_times.append(time.time() - patch_start)

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"TIMING RESULTS (f-AnoGAN)")
    print(f"{'='*50}")
    print(f"Patches processed: {len(patch_times)}")
    print(f"Avg time per patch: {np.mean(patch_times):.3f}s")
    print(f"Min / Max per patch: {np.min(patch_times):.3f}s / {np.max(patch_times):.3f}s")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Estimated time for 3000 patches: {np.mean(patch_times)*3000/60:.1f} min")
    print(f"{'='*50}\n")
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = all_candidates[:top_k]
    rows = [bio_values + [0] for score, bio_values in top_candidates]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'bio{i}' for i in range(1, channels + 1)] + ['presence'])
        writer.writerows(rows)
    print(f"Saved top {len(rows)} pseudo-absence rows (out of {len(all_candidates)} valid pixels) to {output_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--testpath',     required=True)
    p.add_argument('--encoder_path', required=True)   # ADDED
    p.add_argument('--imgsize',    type=int,   default=32)
    p.add_argument('--channels',   type=int,   default=19)
    p.add_argument('--zdims',      type=int,   default=50)
    p.add_argument('--top_k',      type=int,   default=3000)
    p.add_argument('--lrg',        type=float, default=1.3101360595981538e-05)
    p.add_argument('--lrd',        type=float, default=1.7980105256636605e-07)
    p.add_argument('--output',     type=str,   default='grape_pseudo_absences_fanogan.csv')
    args = p.parse_args()
    run(args.testpath, args.imgsize, args.channels, args.zdims,
        args.lrg, args.lrd, args.top_k, args.encoder_path, args.output)