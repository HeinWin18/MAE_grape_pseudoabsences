# patchGan.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, BatchNormalization
from tensorflow.keras.models import Model

eps = 1e-8


def patchgan_discriminator_model(img_size, channels, nf=64):
    """
    Small-context pixel-aligned PatchGAN discriminator (Option B).
    Input: (H, W, C) channels_last. Output: (H, W, 1) logits map (same spatial dims).
    """
    inp = Input((img_size, img_size, channels))
    x = Conv2D(nf, kernel_size=3, strides=1, padding='same')(inp)
    x = LeakyReLU(0.3)(x)

    x = Conv2D(nf, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)

    x = Conv2D(nf, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)

    x = Conv2D(nf, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = LeakyReLU(0.3)(x)

    logits = Conv2D(1, kernel_size=1, strides=1, padding='same')(x)  # (N,H,W,1) logits
    return Model(inputs=inp, outputs=logits, name="PatchDiscriminator")


# -------- mask & masked losses --------
# def build_keep_mask_from_b1(x):
#     b1 = x[..., -1]
#     # print(f"Me: Check how b1 is passed to build_keep_mask_from_b1 in DCGan: {b1}")
#     is_valid = tf.greater(b1, tf.constant(0.0, dtype=b1.dtype))
#
#     not_nan = ~tf.math.is_nan(b1)
#     keep = tf.cast(tf.logical_and(is_valid, not_nan), tf.float32)
#
#     return tf.expand_dims(keep, axis=-1)


# def masked_disc_loss(D, x_real, x_fake, keep_mask):
#     """
#     D: discriminator model (returns logits map)
#     x_real, x_fake: (N,H,W,C)
#     keep_mask: (N,H,W,1) float32 (1 keep, 0 ignore)
#     returns scalar loss (masked, normalized)
#     """
#     logits_real = D(x_real, training=True)  # (N,H,W,1)
#     logits_fake = D(x_fake, training=True)
#
#     loss_real_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real)
#     loss_fake_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake)
#
#     # loss_real_map = tf.square(tf.ones_like(logits_real) - logits_real)
#     # loss_fake_map = tf.square(tf.zeros_like(logits_fake) - logits_fake)
#
#     denom = tf.reduce_sum(keep_mask) + eps
#     loss_real = tf.reduce_sum(loss_real_map * keep_mask) / denom
#     loss_fake = tf.reduce_sum(loss_fake_map * keep_mask) / denom
#
#     # 0.5 is used to slow down d learning, it's a weak control so don't expect much from it
#     return 0.5 * (loss_real + loss_fake)

def masked_disc_loss(D, x_real, x_fake, keep_mask, lambda_gp=10.0):
    logits_real = D(x_real, training=True)
    logits_fake = D(x_fake, training=True)

    denom = tf.reduce_sum(keep_mask) + eps
    loss_real = tf.reduce_sum(-logits_real * keep_mask) / denom  # maximize real scores
    loss_fake = tf.reduce_sum(logits_fake * keep_mask) / denom   # minimize fake scores

    w_loss = loss_real + loss_fake

    gp = gradient_penalty(D, x_real, x_fake, keep_mask)
    return w_loss + lambda_gp * gp

def gradient_penalty(D, x_real, x_fake, keep_mask):
    batch_n = tf.shape(x_real)[0]
    alpha = tf.random.uniform([batch_n, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * x_real + (1 - alpha) * x_fake

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        logits_interp = D(interpolated, training=True)
        masked = tf.reduce_sum(logits_interp * keep_mask) / (tf.reduce_sum(keep_mask) + eps)

    grads = tape.gradient(masked, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + eps)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


# def masked_adv_loss_for_generator(D, x_fake, keep_mask):
#     """
#     Generator adversarial loss using masked discriminator outputs.
#     """
#     logits_fake = D(x_fake, training=True)
#     adv_map = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake)
#
#     # adv_map = tf.square(tf.ones_like(logits_fake)-logits_fake)
#
#     return tf.reduce_sum(adv_map * keep_mask) / (tf.reduce_sum(keep_mask) + eps)

def masked_adv_loss_for_generator(D, x_fake, keep_mask):
    logits_fake = D(x_fake, training=True)

    denom = tf.reduce_sum(keep_mask) + eps
    return tf.reduce_sum(-logits_fake * keep_mask) / denom


def masked_pixel_l1_loss(x_target, x_pred, keep_mask):
    """
    Pixel-wise L1 masked loss. Broadcasts mask across channels.
    """
    mask_c = tf.broadcast_to(keep_mask, tf.shape(x_target))  # (N,H,W,C)
    diff = tf.abs(x_target - x_pred)
    denom = tf.reduce_sum(mask_c) + eps
    return tf.reduce_sum(diff * mask_c) / denom


def masked_pixel_l2_loss(x_target, x_pred, keep_mask):
    """
    """
    mask_c = tf.broadcast_to(keep_mask, tf.shape(x_target))  # (N,H,W,C)
    diffL2 = tf.square(x_target - x_pred)
    denom = tf.reduce_sum(mask_c) + eps
    return tf.reduce_sum(diffL2 * mask_c) / denom


# ======================================================================
# NEW: PATCH-BASED ANOMALY SCORE EXTRACTION
# ======================================================================

def compute_patch_anomaly_scores(x_original, x_generated, discriminator,
                                 mask=None, alpha=0.5, beta=0.5):
    """
    Compute patch-level anomaly scores from original and generated images.

    Args:
        x_original: (N, H, W, C) or (H, W, C) - original input image
        x_generated: (N, H, W, C) or (H, W, C) - generated/reconstructed image
        discriminator: PatchGAN discriminator model
        mask: (N, H, W, 1) or (H, W, 1) or None - optional mask (1=valid, 0=ignore)
        alpha: weight for reconstruction component (default 0.9)
        beta: weight for discriminator component (default 0.1)

    Returns:
        patch_scores: (H, W) numpy array of anomaly scores per pixel/patch
        components: dict with 'residual', 'discriminator', 'combined' score maps
    """
    # Ensure batch dimension
    if len(x_original.shape) == 3:
        x_original = np.expand_dims(x_original, axis=0)
    if len(x_generated.shape) == 3:
        x_generated = np.expand_dims(x_generated, axis=0)

    # Convert to tensors if numpy
    if isinstance(x_original, np.ndarray):
        x_original = tf.constant(x_original, dtype=tf.float32)
    if isinstance(x_generated, np.ndarray):
        x_generated = tf.constant(x_generated, dtype=tf.float32)

    # --- Component 1: Reconstruction residual (pixel-wise) ---
    residual_map = tf.abs(x_original - x_generated)  # (N, H, W, C)

    # ---------------------------------------plotting residual for each channel----------------------------------------
    #
    # import matplotlib.pyplot as plt
    #
    # def plot_residual_channels(residual_map, sample_idx=0):
    #     num_channels = residual_map.shape[-1]
    #     fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
    #
    #     for c in range(num_channels):
    #         axes[c].imshow(residual_map[sample_idx, :, :, c], cmap='hot')
    #         axes[c].set_title(f'Channel {c}')
    #         axes[c].axis('off')
    #
    #     plt.suptitle('Residual Map per Channel')
    #     plt.tight_layout()
    #     plt.show()
    #
    # plot_residual_channels(residual_map)

    # -------------------------------------------------------------------------------------------

    residual_map = tf.reduce_mean(residual_map, axis=-1)  # Average over channels -> (N, H, W)

    # --- Component 2: Discriminator score (realism score) ---
    d_logits = discriminator(x_generated, training=False)  # (N, H, W, 1)
    d_probs = tf.nn.sigmoid(d_logits)  # Convert to probabilities [0,1]
    d_scores = d_probs[..., 0]  # (N, H, W) - higher = more realistic

    # For anomaly detection, we want: high score = more anomalous
    # So invert discriminator: 1 - d_score (high when looks fake)
    d_anomaly = 1.0 - d_scores

    # --- Normalize both components to [0, 1] ---
    # mask_bool = tf.cast(mask, tf.bool)
    mask = tf.expand_dims(mask, axis=0)  # (1, 8, 8)
    print(f"meeeeeeeee: mask shape: {mask.shape}, residual shape: {residual_map.shape}")

    residual_min = tf.reduce_min(residual_map * mask)
    residual_max = tf.reduce_max(residual_map * mask)

    # residual_min = tf.reduce_min(residual_map)
    # residual_max = tf.reduce_max(residual_map)
    residual_norm = (residual_map * mask - residual_min) / (residual_max * mask - residual_min + eps)

    d_min = tf.reduce_min(d_anomaly)
    d_max = tf.reduce_max(d_anomaly)
    d_norm = (d_anomaly - d_min) / (d_max - d_min + eps)

    # --- Combine with weights ---
    combined = alpha * residual_norm + beta * d_norm

    # --- Apply mask if provided ---
    if mask is not None:
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask, axis=0)
        if isinstance(mask, np.ndarray):
            mask = tf.constant(mask, dtype=tf.float32)
        mask_2d = mask[..., 0]  # (N, H, W)

        # Set masked-out regions to 0 (or could use NaN)
        combined = combined * mask_2d
        residual_norm = residual_norm * mask_2d
        d_norm = d_norm * mask_2d

    # Convert to numpy and remove batch dimension
    patch_scores = combined.numpy()[0]  # (H, W)

    components = {
        'residual': residual_norm.numpy()[0],
        'discriminator': d_norm.numpy()[0],
        'combined': patch_scores,
        'mask': mask_2d.numpy()[0] if mask is not None else None
    }

    return patch_scores, components


def compute_patch_statistics(patch_scores, mask=None, percentiles=[50, 75, 90, 95, 99]):
    """
    Compute statistics on patch-level anomaly scores.

    Args:
        patch_scores: (H, W) array of anomaly scores
        mask: (H, W) optional mask (1=valid, 0=ignore)
        percentiles: list of percentiles to compute

    Returns:
        stats: dict with various statistics
    """
    if mask is not None:
        # Only consider valid (masked) pixels
        valid_scores = patch_scores[mask > 0.5]
    else:
        valid_scores = patch_scores.flatten()

    if len(valid_scores) == 0:
        return {'error': 'No valid pixels in mask'}

    stats = {
        'mean': float(np.mean(valid_scores)),
        'std': float(np.std(valid_scores)),
        'min': float(np.min(valid_scores)),
        'max': float(np.max(valid_scores)),
        'median': float(np.median(valid_scores)),
        'percentiles': {f'p{p}': float(np.percentile(valid_scores, p)) for p in percentiles},
        'n_pixels': len(valid_scores)
    }

    return stats


def get_anomalous_pixels(patch_scores, threshold=None, mask=None, top_k=None):
    """
    Identify anomalous pixels based on threshold or top-k.

    Args:
        patch_scores: (H, W) array of anomaly scores
        threshold: float - pixels with score > threshold are anomalous
                   If None, uses 95th percentile
        mask: (H, W) optional mask
        top_k: int - return top k most anomalous pixels (overrides threshold)

    Returns:
        anomaly_mask: (H, W) boolean array
        anomaly_coords: list of (row, col) tuples
        scores: anomaly scores at those coordinates
    """
    if mask is not None:
        valid_mask = mask > 0.5
        masked_scores = patch_scores.copy()
        masked_scores[~valid_mask] = -np.inf  # Ignore masked pixels
    else:
        masked_scores = patch_scores
        valid_mask = np.ones_like(patch_scores, dtype=bool)

    if top_k is not None:
        # Get top-k anomalous pixels
        flat_scores = masked_scores.flatten()
        flat_valid = valid_mask.flatten()

        # Get indices of valid pixels
        valid_indices = np.where(flat_valid)[0]
        valid_scores = flat_scores[valid_indices]

        # Sort and get top k
        top_k_local = min(top_k, len(valid_scores))
        top_indices_local = np.argpartition(valid_scores, -top_k_local)[-top_k_local:]
        top_indices_local = top_indices_local[np.argsort(valid_scores[top_indices_local])[::-1]]

        # Convert back to global indices
        top_indices = valid_indices[top_indices_local]

        # Create mask
        anomaly_mask = np.zeros_like(patch_scores, dtype=bool)
        anomaly_mask.flat[top_indices] = True

    else:
        # Use threshold
        if threshold is None:
            valid_scores = masked_scores[valid_mask]
            threshold = np.percentile(valid_scores, 95)

        anomaly_mask = (masked_scores > threshold) & valid_mask

    # Get coordinates and scores
    anomaly_coords = list(zip(*np.where(anomaly_mask)))
    scores = patch_scores[anomaly_mask]

    return anomaly_mask, anomaly_coords, scores


# ======================================================================
# NEW: VISUALIZATION FUNCTIONS
# ======================================================================

def visualize_patch_scores(patch_scores, mask=None, title="Patch Anomaly Scores",
                           cmap='hot', save_path=None, show=True, figsize=(10, 8)):
    """
    Visualize patch-level anomaly scores as a heatmap.

    Args:
        patch_scores: (H, W) array of anomaly scores
        mask: (H, W) optional mask to show valid regions
        title: plot title
        cmap: colormap (default 'hot')
        save_path: path to save figure (optional)
        show: whether to display the plot
        figsize: figure size

    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Apply mask if provided
    display_scores = patch_scores.copy()
    if mask is not None:
        display_scores[mask < 0.5] = np.nan  # Show masked areas as blank

    # Plot heatmap
    im = ax.imshow(display_scores, cmap=cmap, interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=20)

    # Labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=11)
    ax.set_ylabel('Height (pixels)', fontsize=11)

    # Grid
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] Heatmap -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


def visualize_patch_components(components, mask=None, save_path=None,
                               show=True, figsize=(15, 5)):
    """
    Visualize all components of patch anomaly scores side-by-side.

    Args:
        components: dict from compute_patch_anomaly_scores() with keys:
                    'residual', 'discriminator', 'combined', 'mask'
        mask: (H, W) optional mask (overrides components['mask'])
        save_path: path to save figure
        show: whether to display
        figsize: figure size

    Returns:
        fig, axes: matplotlib objects
    """
    # if mask is None:
    #     mask = components.get('mask', None)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    titles = ['Reconstruction Residual', 'Discriminator Anomaly', 'Combined Score']
    keys = ['residual', 'discriminator', 'combined']

    for ax, title, key in zip(axes, titles, keys):
        data = components[key].copy()

        # if mask is not None:
        #     data[mask < 0.5] = np.nan

        im = ax.imshow(data, cmap='hot', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.grid(False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=15, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] Component visualization -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def visualize_anomalous_regions(original, patch_scores, threshold=None,
                                mask=None, top_k=None, save_path=None,
                                show=True, figsize=(15, 5)):
    """
    Overlay anomalous regions on the original image.

    Args:
        original: (H, W, C) or (H, W) original image
        patch_scores: (H, W) anomaly scores
        threshold: anomaly threshold (if None, uses 95th percentile)
        mask: (H, W) optional mask
        top_k: if provided, highlights top k anomalous pixels
        save_path: path to save figure
        show: whether to display
        figsize: figure size

    Returns:
        fig, axes: matplotlib objects
        anomaly_mask: boolean array of detected anomalies
    """
    # Get anomalous pixels
    anomaly_mask, anomaly_coords, scores = get_anomalous_pixels(
        patch_scores, threshold=threshold, mask=mask, top_k=top_k
    )

    # Prepare original for display
    if len(original.shape) == 3:
        # Multi-channel: take first 3 channels or convert to grayscale
        if original.shape[-1] >= 3:
            display_orig = original[..., :3]
        else:
            display_orig = np.mean(original, axis=-1)
    else:
        display_orig = original

    # Normalize to [0, 1]
    display_orig = (display_orig - display_orig.min()) / (display_orig.max() - display_orig.min() + eps)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Original image
    if len(display_orig.shape) == 2:
        axes[0].imshow(display_orig, cmap='gray')
    else:
        axes[0].imshow(display_orig)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')

    # 2. Anomaly score heatmap
    display_scores = patch_scores.copy()
    if mask is not None:
        display_scores[mask < 0.5] = np.nan
    im = axes[1].imshow(display_scores, cmap='hot', interpolation='nearest')
    axes[1].set_title('Anomaly Scores', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    if len(display_orig.shape) == 2:
        # Convert grayscale to RGB for overlay
        overlay = np.stack([display_orig] * 3, axis=-1)
    else:
        overlay = display_orig.copy()

    # Highlight anomalous regions in red
    overlay_copy = overlay.copy()
    overlay_copy[anomaly_mask, 0] = 1.0  # Red channel
    overlay_copy[anomaly_mask, 1] = 0.0  # Green channel
    overlay_copy[anomaly_mask, 2] = 0.0  # Blue channel

    # Blend
    alpha = 0.5
    blended = alpha * overlay_copy + (1 - alpha) * overlay

    axes[2].imshow(blended)
    axes[2].set_title(f'Detected Anomalies (n={anomaly_mask.sum()})', fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[saved] Anomaly overlay -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes, anomaly_mask


def save_patch_scores_report(patch_scores, components, stats, mask=None,
                             output_dir='outputs', prefix='anomaly'):
    """
    Generate a comprehensive report with all visualizations and statistics.

    Args:
        patch_scores: (H, W) anomaly scores
        components: dict from compute_patch_anomaly_scores()
        stats: dict from compute_patch_statistics()
        mask: (H, W) optional mask
        output_dir: directory to save outputs
        prefix: prefix for filenames

    Returns:
        report_paths: dict of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    report_paths = {}

    # 1. Main heatmap
    heatmap_path = os.path.join(output_dir, f'{prefix}_heatmap.png')
    visualize_patch_scores(patch_scores, mask=mask, save_path=heatmap_path, show=False)
    report_paths['heatmap'] = heatmap_path

    # 2. Component breakdown
    components_path = os.path.join(output_dir, f'{prefix}_components.png')
    visualize_patch_components(components, mask=mask, save_path=components_path, show=False)
    report_paths['components'] = components_path

    # 3. Statistics text file
    stats_path = os.path.join(output_dir, f'{prefix}_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("PATCH ANOMALY SCORE STATISTICS\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Number of pixels: {stats['n_pixels']}\n")
        f.write(f"Mean score: {stats['mean']:.4f}\n")
        f.write(f"Std deviation: {stats['std']:.4f}\n")
        f.write(f"Median: {stats['median']:.4f}\n")
        f.write(f"Min: {stats['min']:.4f}\n")
        f.write(f"Max: {stats['max']:.4f}\n\n")

        f.write("Percentiles:\n")
        for p, val in stats['percentiles'].items():
            f.write(f"  {p}: {val:.4f}\n")

    report_paths['statistics'] = stats_path
    print(f"[saved] Statistics -> {stats_path}")

    # 4. Scores as numpy array
    scores_path = os.path.join(output_dir, f'{prefix}_scores.npy')
    np.save(scores_path, patch_scores)
    report_paths['scores_npy'] = scores_path
    print(f"[saved] Score array -> {scores_path}")

    return report_paths
