# coding:utf-8
import time
import traceback
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os, sys
import math
from tqdm import tqdm
from sklearn.manifold import TSNE

import Playground
import model
import dcgan
import load
import tifffile as tiff
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import mlflow
import optuna

mlflow.set_experiment("AnoGAN")  # groups your runs together

run_name = f"gan_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.start_run(run_name=run_name)  # 1. Log hyperparameters (before training)


def anomaly_detection(test_img, test_mask, args, g=None, d=None, out_path="outputs/similar.tif"):
    """
    Run anomaly detection with patch-level scoring.

    Args:
        test_img: (H, W, C) test image
        test_mask: (H, W) binary mask indicating valid pixels (1 for valid, 0 for no-data)
        args: command line arguments
        g: generator model (optional, will load from weights if None)
        d: discriminator model (optional, will load from weights if None)
        out_path: where to save the generated similar image

    Returns:
        ano_score: overall scalar anomaly score
        patch_scores: (H, W) spatial anomaly score map
        original_hwC: (H, W, C) original image
        similar_hwC: (H, W, C) generated similar image
        residual: (H, W, C) residual map
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print("\n" + "=" * 60)
    print("RUNNING ANOMALY DETECTION")
    print("=" * 60)

    # Run AnoGAN optimization
    print("Building AnoGAN model...")
    anogan_model = model.anomaly_detector(args, g=g, d=d)

    x_in = test_img.reshape(1, args.imgsize, args.imgsize, args.channels)
    mask_in = test_mask.reshape(1, args.imgsize, args.imgsize, 1)

    # print(f"input img shape is : {x_in.shape}")
    # print(f"test mask  is : {test_mask}")
    print("Optimizing latent code (this may take a minute)...")
    ano_score, similar_img = model.compute_anomaly_score(args, anogan_model, x_in, mask_in, iterations=500, d=d)

    # Reshape for processing
    original_hwC = test_img.reshape(args.imgsize, args.imgsize, args.channels)
    similar_hwC = similar_img.reshape(args.imgsize, args.imgsize, args.channels)

    # Compute residual
    residual = (original_hwC - similar_hwC + 2) / 4.0

    # Save similar as TIFF (float32 to preserve range)
    tiff.imwrite(out_path, similar_hwC.astype(np.float32))
    print(f"[saved] Generated similar image -> {out_path}")

    # ===== PATCH-BASED ANOMALY SCORING =====
    print("\nComputing patch-level anomaly scores...")

    from patchGan import (
        compute_patch_anomaly_scores,
        compute_patch_statistics,
        save_patch_scores_report,
        patchgan_discriminator_model
    )

    # Load or use provided discriminator
    if d is None:
        d = patchgan_discriminator_model(args.imgsize, args.channels)
        d.load_weights('/Users/justin/Desktop/Rast-AnoGAN-onAlmond/saved_model/corn/discriminator_20260625_080804.weights.h5')
        print("Loaded discriminator weights")

    # Compute patch scores
    # alpha = weight for reconstruction (higher = focus on pixel differences)
    # beta = weight for discriminator (higher = focus on "fake-looking" areas)
    # b1_band = test_img[..., -1]
    patch_scores, components = compute_patch_anomaly_scores(
        original_hwC,
        similar_hwC,
        d,
        alpha=0.9,  # Prioritize reconstruction for spatial data
        beta=0.1,
        mask=test_mask
    )

    # Compute statistics
    stats = compute_patch_statistics(patch_scores)

    # Generate comprehensive report
    print("Generating patch anomaly report...")
    report_paths = save_patch_scores_report(
        patch_scores,
        components,
        stats,
        output_dir='outputs',
        prefix='patch_anomaly'
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PATCH ANOMALY SCORE STATISTICS")
    print("=" * 60)
    print(f"Overall anomaly score: {ano_score:.4f}")
    print(f"Patch scores - Mean: {stats['mean']:.4f}")
    print(f"             - Std:  {stats['std']:.4f}")
    print(f"             - Min:  {stats['min']:.4f}")
    print(f"             - Max:  {stats['max']:.4f}")
    print(f"             - Median: {stats['median']:.4f}")
    print(f"             - 95th percentile: {stats['percentiles']['p95']:.4f}")
    print(f"             - 99th percentile: {stats['percentiles']['p99']:.4f}")
    print("=" * 60)

    # Identify most anomalous regions
    from patchGan import get_anomalous_pixels
    anom_mask, coords, anom_scores = get_anomalous_pixels(patch_scores, mask=test_mask, top_k=10)

    if len(coords) > 0:
        print(f"\nTop 10 most anomalous pixels:")
        for i, ((r, c), score) in enumerate(zip(coords[:10], anom_scores[:10]), 1):
            print(f"  {i}. Pixel ({r}, {c}): score = {score:.4f}")

    # ---------------------------filtering masked pixels from top 10------------------------------------------------------------------
    #     if len(coords) > 0:
    #         # Filter out masked pixels
    #         valid = [(coord, score) for coord, score in zip(coords, anom_scores) if mask[coord[0], coord[1]]]
    #
    #         print(f"\nTop 10 most anomalous pixels:")
    #         for i, (coord, score) in enumerate(valid[:10], 1):
    #             r, c = coord
    #             print(f"  {i}. Pixel ({r}, {c}): score = {score:.4f}")

    #         ------------------------------------------------------------------------------------------
    print(f"\nReports saved to:")
    for name, path in report_paths.items():
        print(f"  - {name}: {path}")
    print()

    return ano_score, patch_scores, original_hwC, similar_hwC, residual


def analyze_patch_anomalies_detailed(original, generated, patch_scores, discriminator,
                                     output_dir='outputs/detailed_analysis'):
    """
    Perform additional detailed analysis of patch anomalies.
    Optional extended analysis function.
    """
    from patchGan import (
        visualize_patch_scores,
        visualize_patch_components,
        visualize_anomalous_regions,
        compute_patch_anomaly_scores
    )

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("DETAILED PATCH ANALYSIS")
    print("=" * 60)

    # Re-compute with different weighting schemes
    weight_configs = [
        (0.95, 0.05, "reconstruction_focused"),
        (0.5, 0.5, "balanced"),
        (0.1, 0.9, "discriminator_focused")
    ]

    for alpha, beta, label in weight_configs:
        scores, comps = compute_patch_anomaly_scores(
            original, generated, discriminator,
            alpha=alpha, beta=beta
        )

        visualize_patch_scores(
            scores,
            title=f"Anomaly Scores ({label.replace('_', ' ').title()})",
            save_path=f"{output_dir}/heatmap_{label}.png",
            show=False
        )
        print(f"[saved] Heatmap with {label} weighting")

    # Visualize with different thresholds
    thresholds = [0.5, 0.7, 0.9]
    for thresh in thresholds:
        fig, axes, anom_mask = visualize_anomalous_regions(
            original,
            patch_scores,
            threshold=thresh,
            save_path=f"{output_dir}/overlay_thresh_{thresh:.1f}.png",
            show=False
        )
        n_anomalies = anom_mask.sum()
        print(f"[saved] Overlay with threshold={thresh:.1f} ({n_anomalies} anomalous pixels)")

    print("=" * 60 + "\n")

#Parnian's original run function (before refactoring for Optuna and better error handling)
# 
#     """
#     Main execution function.
#     """
#     print("\n" + "=" * 70)
#     print(" AnoGAN with PatchGAN for Spatial Raster Anomaly Detection")
#     print("=" * 70)
#     print(f"Mode: {args.mode}")
#     print(f"Image size: {args.imgsize}x{args.imgsize}")
#     print(f"Channels: {args.channels}")
#     print(f"Latent dims: {args.zdims}")
#     print("=" * 70 + "\n")

#     # ===== LOAD DATA =====
#     print("Loading data...")
#     #Hein's changes
#     # X_train, test_img, valid_masks, test_mask = load.load_raster_data(args.datapath, args.testpath, args.imgsize,
#     #                                                                   args.mode)
    
#     X_train, valid_masks = load.load_raster_data(args.datapath, args.testpath, args.imgsize, args.mode)
#     test_img, test_mask = np.array([]), np.array([])

#     if args.mode == 'train':
#         print(f"Training data shape: {X_train.shape}")
#     if args.mode == 'test' and test_img is not None and len(test_img.shape) > 0:
#         print(f"Test image shape: {test_img.shape}")

#     # ===== INITIALIZE DCGAN/PATCHGAN =====
#     print("\nInitializing PatchGAN model...")
#     DCGAN = dcgan.DCGAN(args)

#     # ===== TRAINING =====
#     if args.mode == 'train':

#         print("\n" + "=" * 60)
#         print("TRAINING PATCHGAN")
#         print("=" * 60)

#         # Validate data
#         if X_train is None or len(X_train) == 0:
#             raise ValueError("No training data loaded! Check your datapath.")

#         # Train using the full training loop
#         # NOTE: dcgan.py needs a train_full() wrapper or we need to call train() in a loop
#         # For now, we'll implement the loop here

#         n_samples = X_train.shape[0]
#         steps_per_epoch = max(1, n_samples // args.batchsize)

#         print(f"Training samples: {n_samples}")
#         print(f"Batch size: {args.batchsize}")
#         print(f"Steps per epoch: {steps_per_epoch}")
#         print(f"Epochs: {args.epoch}")
#         print(f"Total training steps: {args.epoch * steps_per_epoch}\n")

#         d_losses, g_losses = [], []

#         train_start = time.time()
#         for epoch in range(args.epoch):
#             epoch_start = time.time()
#             print(f"\nEpoch {epoch + 1}/{args.epoch}")

#             # Shuffle data
#             np.random.seed(None)
#             indices = np.random.permutation(n_samples)
#             X_shuffled = X_train[indices]

#             masks_shuffled = valid_masks[indices]

#             epoch_d_losses, epoch_g_losses = [], []

#             # Progress bar for this epoch
#             pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")

#             for step in pbar:
#                 batch_start = step * args.batchsize
#                 batch_end = min(batch_start + args.batchsize, n_samples)

#                 if batch_end - batch_start < args.batchsize // 2:
#                     # Skip incomplete batches at end
#                     continue

#                 x_batch = X_shuffled[batch_start:batch_end]
#                 mask_batch = masks_shuffled[batch_start:batch_end]

#                 # Call training step (single batch)
#                 d_loss, g_loss = DCGAN.train(x_batch, mask_batch, l1_weight=0.5, g_steps=args.g_steps)

#                 # # global_step = epoch * len(dataloader) + epoch

#                 epoch_d_losses.append(float(d_loss))
#                 epoch_g_losses.append(float(g_loss))

#                 # Update progress bar
#                 pbar.set_postfix({
#                     'D': f'{d_loss:.4f}',
#                     'G': f'{g_loss:.4f}'
#                 })

#             # Epoch summary
#             avg_d = np.mean(epoch_d_losses)
#             avg_g = np.mean(epoch_g_losses)
#             d_losses.extend(epoch_d_losses)
#             g_losses.extend(epoch_g_losses)
#             mlflow.log_metric("g_loss", avg_g, step=epoch)
#             mlflow.log_metric("d_loss", avg_d, step=epoch)

#             print(f"Epoch {epoch + 1} complete - Avg D: {avg_d:.4f}, Avg G: {avg_g:.4f}")

#             # Save weights periodically
#             if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epoch:
#                 print(f"Saving weights at epoch {epoch + 1}...")
#                 DCGAN.g.save_weights('./saved_model/generator.weights.h5')
#                 DCGAN.d.save_weights('./saved_model/discriminator.weights.h5')
        
#             epoch_time = time.time() - epoch_start
#             elapsed_total = time.time() - train_start
#             remaining = epoch_time * (args.epoch - epoch - 1)
#             print(f"Epoch {epoch + 1} complete - Avg D: {avg_d:.4f}, Avg G: {avg_g:.4f} | "
#             f"Time: {epoch_time:.1f}s | ETA: {remaining/60:.1f}min")
#             mlflow.log_metric("epoch_time", epoch_time, step=epoch)

#         # Final save
#         print("\nTraining complete! Saving final weights...")
#         os.makedirs('./saved_model/', exist_ok=True)
#         DCGAN.g.save('./saved_model/generator.h5')
#         DCGAN.d.save('./saved_model/discriminator.h5')

#         # Plot training losses
#         plt.figure(figsize=(12, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(d_losses, label='Discriminator', alpha=0.7)
#         plt.xlabel('Training Step')
#         plt.ylabel('Loss')
#         plt.title('Discriminator Loss')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         plt.subplot(1, 2, 2)
#         plt.plot(g_losses, label='Generator', alpha=0.7, color='orange')
#         plt.xlabel('Training Step')
#         plt.ylabel('Loss')
#         plt.title('Generator Loss')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         plt.tight_layout()
#         plt.savefig(f'./result/training_losses_{int(time.time())}.png', dpi=150)
#         print(f"[saved] Training loss plot -> ./result/training_losses.png")
#         plt.close()

#         print("\n" + "=" * 60)
#         print("TRAINING COMPLETE!")
#         print("=" * 60)
#         # return float(tf.reduce_mean(g_losses))
#         return float(tf.reduce_mean(g_losses[-steps_per_epoch:]))

#     # ===== TESTING / ANOMALY DETECTION =====
#     if args.mode == 'test':

#         if test_img is None or len(test_img.shape) == 0:
#             print("\nWARNING: No test image provided. Skipping anomaly detection.")
#             print("Use --testpath to specify a test image.")
#             return

#         print("\nRunning anomaly detection on test image...")
#         print(f"Test image shape: {test_img.shape}")

#         # Run anomaly detection with patch-level scoring
#         g = load_model('./saved_model/generator.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
#         d = load_model('./saved_model/discriminator.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
#         score, patch_scores, query, pred, diff = anomaly_detection(
#             test_img, test_mask, args, g=g, d=d)

#         # Standard visualization (from Playground.py)
#         print("\nGenerating standard visualizations...")
#         Playground.illustrate(query, pred, diff, base="outputs/")

#         # Optional: Extended detailed analysis
#         if args.detailed_analysis:
#             analyze_patch_anomalies_detailed(
#                 query, pred, patch_scores, DCGAN.d,
#                 output_dir='outputs/detailed_analysis'
#             )

#     print("\n" + "=" * 70)
#     print(" EXECUTION COMPLETE")
#     print("=" * 70 + "\n")

#Version 1 - run code with time added
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import mlflow
from tensorflow.keras.models import load_model

# def run(args):
#     # ===== SETUP =====
#     X_train, valid_masks = load.load_raster_data(args.datapath, args.testpath, args.imgsize, args.mode)
#     test_img, test_mask = np.array([]), np.array([])
#     DCGAN = dcgan.DCGAN(args)

#So even in test mode, test_img was immediately overwritten with an empty array. When the test block ran, it saw an empty test_img and silently returned without doing anything.
#The fix: We split the loading by mode. Train mode loads training data. Test mode loads test data. They no longer overwrite each other.
def run(args):

    # ============================================================
    # SETUP
    # ============================================================

    if args.mode == 'train':

        X_train, valid_masks = load.load_raster_data(
            args.datapath,
            args.testpath,
            args.imgsize,
            args.mode
        )

        X_train = np.clip(X_train, -3, 3) / 3.0

        test_img, test_mask = np.array([]), np.array([])

    elif args.mode == 'test':

        all_test, all_masks = load.load_raster_data(
            args.datapath,
            args.testpath,
            args.imgsize,
            args.mode
        )

        test_img = all_test[0]
        test_mask = all_masks[0]

        X_train, valid_masks = np.array([]), np.array([])


    # ============================================================
    # CREATE DCGAN
    # ============================================================

    DCGAN = dcgan.DCGAN(args)


    # ============================================================
    # TRAIN GAN
    # ============================================================

    if args.mode == 'train':

        n_samples = X_train.shape[0]
        steps_per_epoch = max(1, n_samples // args.batchsize)

        d_losses, g_losses = [], []

        train_start = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for epoch in range(args.epoch):

            epoch_start = time.time()

            print(
                f"\nEpoch {epoch + 1}/{args.epoch}"
            )

            np.random.seed(None)

            indices = np.random.permutation(n_samples)

            X_shuffled = X_train[indices]
            masks_shuffled = valid_masks[indices]

            epoch_d_losses, epoch_g_losses = [], []

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch {epoch + 1}"
            )

            for step in pbar:

                batch_start = step * args.batchsize
                batch_end = min(
                    batch_start + args.batchsize,
                    n_samples
                )

                if batch_end - batch_start < args.batchsize // 2:
                    continue

                x_batch = X_shuffled[
                    batch_start:batch_end
                ]

                mask_batch = masks_shuffled[
                    batch_start:batch_end
                ]

                d_loss, g_loss = DCGAN.train(
                    x_batch,
                    mask_batch,
                    l1_weight=0.5,
                    g_steps=args.g_steps
                )

                epoch_d_losses.append(
                    float(d_loss)
                )

                epoch_g_losses.append(
                    float(g_loss)
                )

                pbar.set_postfix({
                    'D': f'{d_loss:.4f}',
                    'G': f'{g_loss:.4f}'
                })

            avg_d = np.mean(epoch_d_losses)
            avg_g = np.mean(epoch_g_losses)

            d_losses.extend(epoch_d_losses)
            g_losses.extend(epoch_g_losses)

            mlflow.log_metric(
                "g_loss",
                avg_g,
                step=epoch
            )

            mlflow.log_metric(
                "d_loss",
                avg_d,
                step=epoch
            )

            epoch_time = time.time() - epoch_start

            remaining = (
                epoch_time *
                (args.epoch - epoch - 1)
            )

            print(
                f"Epoch {epoch + 1} complete - "
                f"Avg D: {avg_d:.4f}, "
                f"Avg G: {avg_g:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"ETA: {remaining/60:.1f}min"
            )

            mlflow.log_metric(
                "epoch_time",
                epoch_time,
                step=epoch
            )

            if (
                (epoch + 1) % 10 == 0
                or (epoch + 1) == args.epoch
            ):

                os.makedirs(
                    args.output_dir,
                    exist_ok=True
                )

                DCGAN.g.save_weights(
                    os.path.join(
                        args.output_dir,
                        f'generator_{timestamp}.weights.h5'
                    )
                )

                DCGAN.d.save_weights(
                    os.path.join(
                        args.output_dir,
                        f'discriminator_{timestamp}.weights.h5'
                    )
                )

        total_time = time.time() - train_start

        mlflow.log_metric(
            "total_train_time_sec",
            total_time
        )

        mlflow.log_metric(
            "avg_epoch_time_sec",
            total_time / args.epoch
        )

        DCGAN.g.save(
            os.path.join(
                args.output_dir,
                f'generator{timestamp}.h5'
            )
        )

        DCGAN.d.save(
            os.path.join(
                args.output_dir,
                f'discriminator{timestamp}.h5'
            )
        )

        return float(
            tf.reduce_mean(
                g_losses[-steps_per_epoch:]
            )
        )


    # ============================================================
    # TEST
    # ============================================================

    elif args.mode == 'test':

        if test_img is None or len(test_img.shape) == 0:
            return

        g = load_model(
            '/Users/justin/Desktop/Rast-AnoGAN-onAlmond/'
            'saved_model/corn/'
            'generator_20260625_080804.weights.h5',
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        d = load_model(
            '/Users/justin/Desktop/Rast-AnoGAN-onAlmond/'
            'saved_model/corn/'
            'discriminator_20260625_080804.weights.h5',
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        score, patch_scores, query, pred, diff = anomaly_detection(
            test_img,
            test_mask,
            args,
            g=g,
            d=d
        )

        Playground.illustrate(
            query,
            pred,
            diff,
            base="outputs/"
        )

        if getattr(
            args,
            'detailed_analysis',
            False
        ):

            analyze_patch_anomalies_detailed(
                query,
                pred,
                patch_scores,
                DCGAN.d,
                output_dir='outputs/detailed_analysis'
            )


    # ============================================================
    # TRAIN ENCODER
    # ============================================================

    elif args.mode == 'train_encoder':

        print("\n" + "=" * 60)
        print("TRAINING f-AnoGAN ENCODER")
        print("=" * 60)

        print("\nData path:")
        print(args.datapath)

        print("\nGenerator:")
        print(args.generator_path)

        print("\nDiscriminator:")
        print(args.discriminator_path)

        print("\nEncoder epochs:")
        print(args.encoder_epochs)

        # --------------------------------------------------------
        # Load training data
        # --------------------------------------------------------

        X_train, valid_masks = load.load_raster_data(
            args.datapath,
            args.testpath,
            args.imgsize,
            'train'
        )

        print(
            f"\nTraining data shape: "
            f"{X_train.shape}"
        )

        # Same normalization used previously
        X_train = np.clip(
            X_train,
            -3,
            3
        ) / 3.0

        # --------------------------------------------------------
        # Load pretrained generator
        # --------------------------------------------------------

        print("\nLoading pretrained generator...")

        g = load_model(
            args.generator_path,
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        print("Generator loaded successfully.")

        # --------------------------------------------------------
        # Load pretrained discriminator
        # --------------------------------------------------------

        print("\nLoading pretrained discriminator...")

        d = load_model(
            args.discriminator_path,
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        print("Discriminator loaded successfully.")

        print("\nLoaded existing GAN weights.")
        print("Skipping GAN training.")

        # --------------------------------------------------------
        # Train encoder
        # --------------------------------------------------------

        print(
            f"\nStarting encoder training "
            f"for {args.encoder_epochs} epochs..."
        )

        os.makedirs(
            args.output_dir,
            exist_ok=True
        )

        encoder = model.train_encoder(
            args,
            g,
            d,
            X_train,
            valid_masks,
            epochs=args.encoder_epochs,
            save_path=args.output_dir
        )

        print(
            "\nEncoder training complete."
        )

        print(
            f"Encoder saved in: "
            f"{args.output_dir}"
        )


    # ============================================================
    # TRAIN UNMASKED ENCODER
    # ============================================================

    elif args.mode == 'train_encoder_unmasked':

        X_train, valid_masks = load.load_raster_data(
            args.datapath,
            args.testpath,
            args.imgsize,
            'train'
        )

        X_train = np.clip(
            X_train,
            -3,
            3
        ) / 3.0

        g = load_model(
            '/Users/justin/Desktop/Rast-AnoGAN-onAlmond/'
            'saved_model/corn/'
            'generator20260625_080804.h5',
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        d = load_model(
            '/Users/justin/Desktop/Rast-AnoGAN-onAlmond/'
            'saved_model/corn/'
            'discriminator20260625_080804.h5',
            custom_objects={
                'mse':
                tf.keras.losses.MeanSquaredError()
            }
        )

        print(
            "Loaded existing GAN weights — "
            "skipping GAN training."
        )

        encoder = model.train_encoder_new_loss(
            args,
            g,
            d,
            X_train,
            valid_masks,
            epochs=500,
            save_path='./saved_model/'
        )

        print(
            "Encoder training complete."
        )
        
def validate_args(args):
    """
    Validate command line arguments.
    """
    # Check imgsize is compatible with generator architecture
    if args.imgsize % 4 != 0:
        raise ValueError(
            f"imgsize must be divisible by 4 for current generator architecture. "
            f"Got {args.imgsize}. Try: 4, 8, 12, 16, 20, 24, 28, 32, etc."
        )

    if args.mode == 'train_encoder':
        if not args.datapath:
            raise ValueError(
                "--datapath is required for encoder training"
            )
    
        if not args.generator_path:
            raise ValueError(
                "--generator_path is required for encoder training"
            )
    
        if not args.discriminator_path:
            raise ValueError(
                "--discriminator_path is required for encoder training"
            )
    
        if not os.path.exists(args.generator_path):
            raise FileNotFoundError(
                f"Generator not found: {args.generator_path}"
            )
    
        if not os.path.exists(args.discriminator_path):
            raise FileNotFoundError(
                f"Discriminator not found: {args.discriminator_path}"
            )

    # Recommend power-of-2 sizes for best results
    if args.imgsize not in [4, 8, 16, 32, 64, 128]:
        print(f"WARNING: imgsize={args.imgsize} is not a power of 2.")
        print(f"         For optimal results, use: 4, 8, 16, 32, 64, or 128")

    # Check channels
    if args.channels < 1:
        raise ValueError(f"channels must be positive, got {args.channels}")

    # Check mode
    if args.mode not in ['train', 'test', 'train_encoder', 'train_encoder_unmasked']:
        raise ValueError(f"mode must be 'train', 'test', 'train_encoder', or 'train_encoder_unmasked', got {args.mode}")

    # Check required paths
    if args.mode == 'train' and not args.datapath:
        raise ValueError("--datapath is required for training mode")

    if args.mode == 'test' and not args.testpath:
        print("WARNING: --testpath not provided for test mode")
        print("         Will skip anomaly detection")

    # Check epoch and batch size
    if args.epoch < 1:
        raise ValueError(f"epoch must be positive, got {args.epoch}")
    if args.batchsize < 1:
        raise ValueError(f"batchsize must be positive, got {args.batchsize}")


def main(config=None):
    if config is not None:
        # Called from Optuna objective — use config dict directly
        class Args:
            pass

        args = Args()
        args.mode = 'train'
        args.lrg = config.get('lrg', 1e-3)
        args.lrd = config.get('lrd', 1e-4)
        args.batchsize = config.get('batch_size', 64)
        args.g_steps = config.get('g_steps', 1)
        args.zdims = config.get('z_dim', 100)
        # Set defaults for remaining args
        args.datapath = config.get('datapath', 'AlmondData/TrainingPatches-4')
        args.testpath = config.get('testpath', 'AlmondData/TrainingPatches-4/patch_00000.tif')
        args.epoch = config.get('epoch', 500)
        args.imgsize = config.get('imgsize', 4)
        args.channels = config.get('channels', 20)
        args.detailed_analysis = config.get('detailed_analysis', False)
        args.label_idx = config.get('label_idx', 1)
        args.img_idx = config.get('img_idx', 14)
    else:
        parser = argparse.ArgumentParser(
            description='AnoGAN with PatchGAN for Spatial Raster Anomaly Detection',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples:
      # Training
      python main.py --mode train --datapath train_patches/ --epoch 100 --batchsize 32
    
      # Testing
      python main.py --mode test --testpath test_patch.tif
    
      # Custom size
      python main.py --mode train --datapath data/ --imgsize 16 --channels 10
            """
        )

        # Data paths
        parser.add_argument(
                '--output_dir',
                type=str,
                default='./saved_model',
                help='Directory for saving trained models and results'
            )
        parser.add_argument(
            '--generator_path',
            type=str,
            default=None,
            help='Path to trained f-AnoGAN generator .h5 file'
        )
        
        parser.add_argument(
            '--discriminator_path',
            type=str,
            default=None,
            help='Path to trained f-AnoGAN discriminator .h5 file'
        )
        
        parser.add_argument(
            '--encoder_epochs',
            type=int,
            default=500,
            help='Number of epochs for encoder training'
        )
        parser.add_argument('--datapath', '-d', type=str,
                            help='Path to training data directory (contains .tif patches)')
        parser.add_argument('--testpath', '-p', type=str,
                            help='Path to test image (.tif file)')

        # Training parameters
        parser.add_argument('--mode', '-m', type=str, default='train',
                    choices=['train', 'test', 'train_encoder', 'train_encoder_unmasked'],
                    help='Mode: train, test, or train_encoder')
        parser.add_argument('--epoch', '-e', type=int, default=500,
                            help='Number of training epochs (default: 500)')
        parser.add_argument('--batchsize', '-b', type=int, default=64,
                            help='Batch size for training (default: 32)')

        # Model architecture
        parser.add_argument('--imgsize', type=int, default=32,
                            help='Image size H=W (must be divisible by 4, default: 8)')
        parser.add_argument('--channels', type=int, default=20,
                            help='Number of image channels (default: 20)')
        parser.add_argument('--zdims', type=int, default=100,
                            help='Latent space dimensions (default: 100)')
    

        # Analysis options
        parser.add_argument('--detailed_analysis', action='store_true',
                            help='Run extended detailed analysis with multiple visualizations')

        # Legacy parameters (kept for compatibility)
        parser.add_argument('--img_idx', type=int, default=14,
                            help='(legacy parameter, not used)')
        parser.add_argument('--label_idx', type=int, default=1,
                            help='(legacy parameter, not used)')
        parser.add_argument('--lrg', type=float, default=0.001,
                            help='(generator learning rate)')
        parser.add_argument('--lrd', type=float, default=0.001,
                            help='(discriminator learning rate)')
        parser.add_argument('--g_steps', type=int, default=1,
                    help='Generator steps per discriminator step')
    

        args = parser.parse_args()

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"\nERROR: {e}\n")
        parser.print_help()
        raise ValueError(f"validate_args failed: {e}")  # NO sys.exit
        # sys.exit(1)

    # Run main execution
    try:
        return run(args)
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def objective(trial):
    config = {
        "mode": "train",
        "epoch": 50,
        "datapath": 'AlmondData/TrainingPatches_4.1',
        # "datapath": 'AlmondData/SmallSet',
        "testpath": 'AlmondData/TrainingPatches-4/patch_00000.tif',
        "lrg": trial.suggest_float("lrg", 1e-4, 1e-2, log=True),
        "lrd": trial.suggest_float("lrd", 1e-5, 1e-3, log=True),
        # "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "batch_size": trial.suggest_categorical("batch_size", [8]),
        "imgsize": 4,
        "channels": 20,
        "detailed_analysis": False,
        "img_idx": 14,
        "label_idx": 1,
        "g_steps": trial.suggest_int("g_steps", 4, 8),
        "z_dim": trial.suggest_categorical("z_dim", [50, 100, 200]),
    }

    mlflow.end_run()  # ← end any previously active run first
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "lrg": config["lrg"],
            "lrd": config["lrd"],
            "batch_size": config["batch_size"],
            "g_steps": config["g_steps"],
            "z_dim": config["z_dim"],
        })
    return main(config)


if __name__ == '__main__':
    main()
    # mlflow.set_experiment("anogan_optuna")

    # study = optuna.create_study(direction="minimize")
    # try:
    #     study.optimize(objective, n_trials=50, catch=(Exception,))
    # except Exception:
    #     traceback.print_exc()

    # # Results
    # if study.best_trials:  # ← protect against all trials failing
    #     print("Best trial:")
    #     print(f"  Value: {study.best_trial.value}")
    #     print(f"  Params: {study.best_trial.params}")
    # else:
    #     print("No trials completed successfully")
