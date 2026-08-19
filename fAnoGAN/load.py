# coding:utf-8

import numpy as np
import os, sys
import glob
import csv
from pathlib import Path

try:
    from keras.preprocessing.image import load_img, img_to_array
except ImportError:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
import rasterio
from rasterio.errors import RasterioIOError


# ======================================================================
# RASTER DATA LOADING (MAIN FUNCTION)
# ======================================================================

def load_raster_data(dataset_path=None, testset_path=None, img_size=32, mode="train",
                     expect_channels=None, skip_mismatch=True, nan_fill=0.0,
                     scale_mode="zscore", max_samples=None, shuffle=True,
                     random_seed=42, augment=False):
    """
    Load multi-band .tif patches for training/testing with enhanced features.

    Args:
        dataset_path (str): folder containing *.tif patches of training data
        testset_path (str): folder containing *.tif patches of test data
        img_size (int): expected H=W size for patches
        mode (str): 'train' or 'test'
        expect_channels (int or None): expected channel count. If None, inferred from first file.
        skip_mismatch (bool): skip patches not exactly img_size or channel count
        nan_fill (float): value to replace NaNs with
        scale_mode (str): 'zscore' | 'minus1_1' | 'minmax' | 'none'
        max_samples (int or None): limit number of training samples (for testing)
        shuffle (bool): shuffle training data before loading
        random_seed (int): random seed for reproducibility
        augment (bool): apply data augmentation to training data

    Returns:
        raster_files (np.ndarray): Training data (N, H, W, C)
        X_test (np.ndarray): Test data (H, W, C) or empty array

    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        ValueError: If no valid patches found or test patch invalid
    """

    raster_files = []
    masks = []
    files = ""

    # --- Load training patches ---
    print(f"\n{'=' * 60}")
    print(f"LOADING RASTER DATA")
    print(f"{'=' * 60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Expected size: {img_size}x{img_size}")
    print(f"Scale mode: {scale_mode}")

    if mode == 'train':
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        files = sorted(glob.glob(os.path.join(dataset_path, "*.tif")))
    else:
        if not os.path.exists(testset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        files = sorted(glob.glob(os.path.join(testset_path, "*.tif")))



    if not files:
        raise ValueError(f"No .tif files found ")

    print(f"Found {len(files)} .tif files")

    # Shuffle if requested
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(files)

    # Limit samples if requested
    if max_samples is not None:
        files = files[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Load patches
    skipped = 0
    errors = 0
    for fp in files:
        try:
            with rasterio.open(fp) as src:
                # Read to array (bands, H, W)
                arr = src.read()  # shape: (C, H, W)
                # Move to (H, W, C)
                arr = np.moveaxis(arr, 0, -1)

                # Size/channel checks
                H, W, C = arr.shape
                # print(f"Me: attempt to see the arr: {arr.shape}")

                # Infer expected channels from first valid file
                if expect_channels is None:
                    expect_channels = C
                    print(f"Inferred {expect_channels} channels from first file")

                # Skip mismatched patches
                if skip_mismatch and (H != img_size or W != img_size or C != expect_channels):
                    skipped += 1
                    continue

                # NaN handling
                # if np.isnan(arr).any():
                #     n_nans = np.isnan(arr).sum()
                #     arr = np.nan_to_num(arr, nan=nan_fill)
                #     if n_nans > arr.size * 0.2:  # Warn if >10% NaNs
                #         print(f"  Warning: {os.path.basename(fp)} has {n_nans} NaNs ({n_nans / arr.size * 100:.1f}%)")

                # b1 = arr[:, :, -1]
                # valid_mask = (b1 == 1.0) & ~np.isnan(b1)
                # masks.append(valid_mask.astype(np.float32))

                # if n_valid == 0:
                #     print("⚠️ Warning: No valid pixels found! Disabling masking.")
                #     use_masking = False
                print(f"file name : {os.path.basename(fp)}")
                # raster_files.append(arr.astype(np.float32))
                raster_files.append(arr[:, :, :-1].astype(np.float32))
                if mode == 'train':
                    masks.append(np.nan_to_num(arr[:, :, -1], nan=0.0).astype(np.float32))
                elif mode == 'test':
                    mask = ~np.isnan(arr[:, :, -1])

                    print(f"{arr[:, :, -1]}")
                    print(f"Meeeeeeeeeeeeeeee: want to see what happens to nans : {mask.astype(np.float32)}")

                    masks.append(mask.astype(np.float32))



        except RasterioIOError as e:
            errors += 1
            print(f"  Error reading {os.path.basename(fp)}: {e}")
        except Exception as e:
            errors += 1
            print(f"  Unexpected error with {os.path.basename(fp)}: {e}")

    # Summary
    print(f"\nLoading summary:")
    print(f"  Loaded: {len(raster_files)} patches")
    print(f"  Skipped (size/channel mismatch): {skipped}")
    print(f"  Errors: {errors}")

    if not raster_files:
        raise ValueError(
            f"No usable training patches found.\n"
            f"  Expected: {img_size}x{img_size}x{expect_channels}\n"
            f"  Skipped: {skipped}, Errors: {errors}"
        )

    raster_files = np.stack(raster_files, axis=0)  # (N, H, W, C)
    masks = np.stack(masks, axis=0)  # (N, H, W)

    #Hein's changes -
    raster_files = np.nan_to_num(raster_files, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Normalization ---
    print(f"\nNormalizing with '{scale_mode}' method...")

    if scale_mode == "minus1_1":
        # For uint8-style data in [0,255]
        raster_files = (raster_files - 127.5) / 127.5
        print(f"  Range after normalization: [{raster_files.min():.3f}, {raster_files.max():.3f}]")

    elif scale_mode == "zscore":
        # Per-band z-score across the dataset
        mean = raster_files.mean(axis=(0, 1, 2), keepdims=True)
        std = raster_files.std(axis=(0, 1, 2), keepdims=True) + 1e-7
        raster_files = (raster_files - mean) / std
        print(f"  Range after normalization: [{raster_files.min():.3f}, {raster_files.max():.3f}]")
        print(f"  Per-channel mean: min={mean.min():.3f}, max={mean.max():.3f}")
        print(f"  Per-channel std: min={std.min():.3f}, max={std.max():.3f}")

    elif scale_mode == "minmax":
        # Min-max scaling to [-1, 1]
        min_val = raster_files.min(axis=(0, 1, 2), keepdims=True)
        max_val = raster_files.max(axis=(0, 1, 2), keepdims=True)
        raster_files = 2 * (raster_files - min_val) / (max_val - min_val + 1e-7) - 1
        print(f"  Range after normalization: [{raster_files.min():.3f}, {raster_files.max():.3f}]")

    elif scale_mode == "none":
        print("  No normalization applied")
    else:
        print(f"  Warning: Unknown scale_mode '{scale_mode}', no normalization applied")

    # --- Data augmentation ---
    if augment and mode == 'train':
        print("\nApplying data augmentation...")
        raster_files = augment_raster_data(raster_files)
        print(f"  Augmented shape: {raster_files.shape}")

    # # --- Load test patch ---
    # if mode == 'test' and test_tif:
    #     print(f"\nLoading test patch: {test_tif}")
    #
    #     try:
    #         with rasterio.open(test_tif) as src:
    #             t = src.read()
    #             t = np.moveaxis(t, 0, -1)  # (H, W, C)
    #
    #             # Validate shape
    #             if skip_mismatch and (
    #                     t.shape[0] != img_size or t.shape[1] != img_size or t.shape[2] != expect_channels):
    #                 raise ValueError(
    #                     f"Test patch shape mismatch:\n"
    #                     f"  Expected: ({img_size}, {img_size}, {expect_channels})\n"
    #                     f"  Got: {t.shape}"
    #                 )
    #
    #             # NaN handling
    #             if np.isnan(t).any():
    #                 n_nans = np.isnan(t).sum()
    #                 print(f"  Warning: Test patch has {n_nans} NaNs ({n_nans / t.size * 100:.1f}%)")
    #                 t = np.nan_to_num(t, nan=nan_fill)
    #
    #             t = t.astype(np.float32)
    #             test_mask = t[:, :, -1].astype(np.float32)
    #
    #             # Apply same normalization
    #             if scale_mode == "minus1_1":
    #                 t = (t - 127.5) / 127.5
    #             elif scale_mode == "zscore":
    #                 # Use training statistics
    #                 mean = t.mean(axis=(0, 1, 2), keepdims=True)
    #                 std = t.std(axis=(0, 1, 2), keepdims=True) + 1e-7
    #                 t = (t - mean) / std
    #                 # t = (t - mean.squeeze()) / std.squeeze()
    #
    #             elif scale_mode == "minmax":
    #                 t = 2 * (t - min_val.squeeze()) / (max_val.squeeze() - min_val.squeeze() + 1e-7) - 1
    #
    #             X_test = t  # (H, W, C)
    #             print(f"Test shape: {X_test.shape} (H, W, C)")
    #
    #     except Exception as e:
    #         raise ValueError(f"Error loading test patch: {e}")
    # else:
    #     X_test = np.array([])
    #     test_mask = np.array([])
    # print(f"{'=' * 60}\n")

    return raster_files, masks


# ======================================================================
# DATA AUGMENTATION
# ======================================================================

def augment_raster_data(X, augmentation_factor=2):
    """
    Apply data augmentation to raster patches.

    Augmentations:
        - 90°, 180°, 270° rotations
        - Horizontal flip
        - Vertical flip

    Args:
        X: (N, H, W, C) input data
        augmentation_factor: how many times to augment (1-4)

    Returns:
        Augmented data (N*augmentation_factor, H, W, C)
    """
    augmented = [X]

    if augmentation_factor >= 2:
        # Horizontal flip
        augmented.append(np.flip(X, axis=2))

    if augmentation_factor >= 3:
        # Vertical flip
        augmented.append(np.flip(X, axis=1))

    if augmentation_factor >= 4:
        # 180° rotation
        augmented.append(np.rot90(X, k=2, axes=(1, 2)))

    return np.concatenate(augmented, axis=0)


# ======================================================================
# DATA VALIDATION UTILITIES
# ======================================================================

def validate_raster_data(X, img_size, channels, verbose=True):
    """
    Validate loaded raster data.

    Args:
        X: Data array to validate
        img_size: Expected spatial size
        channels: Expected number of channels
        verbose: Print validation results

    Returns:
        is_valid: Boolean
        issues: List of issue strings
    """
    issues = []

    # Check shape
    if len(X.shape) != 32:
        issues.append(f"Expected 4D array, got {len(X.shape)}D")
    elif X.shape[1] != img_size or X.shape[2] != img_size:
        issues.append(f"Expected spatial size {img_size}x{img_size}, got {X.shape[1]}x{X.shape[2]}")
    elif X.shape[3] != channels:
        issues.append(f"Expected {channels} channels, got {X.shape[3]}")

    # Check for NaNs
    if np.isnan(X).any():
        n_nans = np.isnan(X).sum()
        issues.append(f"Contains {n_nans} NaN values ({n_nans / X.size * 100:.2f}%)")

    # Check for Infs
    if np.isinf(X).any():
        n_infs = np.isinf(X).sum()
        issues.append(f"Contains {n_infs} Inf values ({n_infs / X.size * 100:.2f}%)")

    # Check value range
    min_val, max_val = X.min(), X.max()
    if min_val == max_val:
        issues.append(f"All values are constant: {min_val}")

    is_valid = len(issues) == 0

    if verbose:
        if is_valid:
            print(f"✓ Data validation passed")
            print(f"  Shape: {X.shape}")
            print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
            print(f"  Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        else:
            print(f"✗ Data validation failed:")
            for issue in issues:
                print(f"  - {issue}")

    return is_valid, issues


def analyze_raster_statistics(X, channel_names=None):
    """
    Compute and display detailed statistics for raster data.

    Args:
        X: (N, H, W, C) raster data
        channel_names: Optional list of channel names

    Returns:
        stats: Dict with statistics
    """
    stats = {
        'n_samples': X.shape[0],
        'shape': X.shape,
        'dtype': X.dtype,
        'global_min': float(X.min()),
        'global_max': float(X.max()),
        'global_mean': float(X.mean()),
        'global_std': float(X.std()),
        'per_channel': []
    }

    print("\n" + "=" * 60)
    print("RASTER DATA STATISTICS")
    print("=" * 60)
    print(f"Shape: {X.shape} (N, H, W, C)")
    print(f"Dtype: {X.dtype}")
    print(f"Total pixels: {X.size:,}")
    print(f"\nGlobal statistics:")
    print(f"  Min: {stats['global_min']:.4f}")
    print(f"  Max: {stats['global_max']:.4f}")
    print(f"  Mean: {stats['global_mean']:.4f}")
    print(f"  Std: {stats['global_std']:.4f}")

    print(f"\nPer-channel statistics:")
    for c in range(X.shape[-1]):
        channel_data = X[..., c]
        ch_stats = {
            'min': float(channel_data.min()),
            'max': float(channel_data.max()),
            'mean': float(channel_data.mean()),
            'std': float(channel_data.std())
        }
        stats['per_channel'].append(ch_stats)

        ch_name = channel_names[c] if channel_names else f"Channel {c + 1}"
        print(f"  {ch_name:20s}: min={ch_stats['min']:7.3f}, max={ch_stats['max']:7.3f}, "
              f"mean={ch_stats['mean']:7.3f}, std={ch_stats['std']:6.3f}")

    print("=" * 60 + "\n")

    return stats


# ======================================================================
# TRAIN/VAL SPLIT UTILITY
# ======================================================================

def split_train_val(X, val_split=0.1, shuffle=True, random_seed=42):
    """
    Split data into train and validation sets.

    Args:
        X: (N, H, W, C) data array
        val_split: Fraction for validation (0.0 to 1.0)
        shuffle: Shuffle before splitting
        random_seed: Random seed for reproducibility

    Returns:
        X_train, X_val
    """
    n_samples = X.shape[0]
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    if shuffle:
        indices = np.random.RandomState(random_seed).permutation(n_samples)
        X = X[indices]

    X_train = X[:n_train]
    X_val = X[n_train:]

    print(f"Split: {n_train} train, {n_val} val ({val_split * 100:.1f}%)")

    return X_train, X_val


# ======================================================================
# IMAGE DATA LOADING (for non-raster images)
# ======================================================================

def load_image_data(dataset_path, test_img, img_size, mode, channels=1):
    """
    Load regular image files (JPEG, PNG, etc.).

    Args:
        dataset_path: Directory containing images
        test_img: Path to test image
        img_size: Target size (will resize)
        mode: 'train' or 'test'
        channels: 1 for grayscale, 3 for RGB

    Returns:
        X_train, X_test
    """
    X_train = []
    X_test = []

    print(f"\nLoading image data from: {dataset_path}")

    # Support multiple formats
    patterns = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
    train_image_list = []
    for pattern in patterns:
        train_image_list.extend(glob.glob(os.path.join(dataset_path, pattern)))

    if not train_image_list:
        raise FileNotFoundError(f"No image files found in {dataset_path}")

    print(f"Found {len(train_image_list)} images")

    grayscale = (channels == 1)

    for img_path in train_image_list:
        try:
            img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale' if grayscale else 'rgb')
            imgarray = img_to_array(img)
            X_train.append(imgarray)
        except Exception as e:
            print(f"  Error loading {os.path.basename(img_path)}: {e}")

    X_train = np.array(X_train).astype(np.float32)
    X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]

    print(f"Train shape: {X_train.shape}")

    if mode == 'test' and test_img:
        try:
            test_img = load_img(test_img, target_size=(img_size, img_size),
                                color_mode='grayscale' if grayscale else 'rgb')
            test_imgarray = img_to_array(test_img)
            X_test = np.array(test_imgarray).astype(np.float32)
            X_test = (X_test - 127.5) / 127.5
            print(f"Test shape: {X_test.shape}")
        except Exception as e:
            print(f"Error loading test image: {e}")
            X_test = np.array([])

    return X_train, X_test


# ======================================================================
# CSV DATA LOADING
# ======================================================================

def load_csv_data(dataset_path, img_size):
    """
    Load data from CSV file (for MNIST-style datasets).

    Args:
        dataset_path: Path to CSV file
        img_size: Expected image size (H=W)

    Returns:
        X_train, X_test, X_test_original, Y_test
    """
    X_ = []
    X_data = []
    Y_data = []

    print(f"Loading CSV data from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            Y_data.append(int(row[-1]))
            X_ = list(map(int, row[:img_size * img_size]))
            X_ = np.array(X_).reshape(img_size, img_size)
            X_data.append(X_)

        train_len = int(len(X_) * 0.9)
        validation_len = len(X_) - train_len

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_data, Y_data, test_size=validation_len, random_state=42
        )

        X_train = (np.array(X_train).astype(np.float32) - 127.5) / 127.5
        X_test = (np.array(X_test).astype(np.float32) - 127.5) / 127.5
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        X_train = X_train[:, :, :, None]
        X_test = X_test[:, :, :, None]

        X_test_original = X_test.copy()

        print(f"Loaded: {len(X_train)} train, {len(X_test)} test")
        print(f"Label distribution: {np.bincount(Y_train)}")

        # Filter for single class (anomaly detection setup)
        X_train = X_train[Y_train == 1]
        X_test = X_test[Y_test == 1]

        return X_train, X_test, X_test_original, Y_test


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def save_normalization_stats(output_path, mean, std, scale_mode="zscore"):
    """
    Save normalization statistics for later use.

    Args:
        output_path: Where to save (e.g., './stats/norm_stats.npz')
        mean: Mean values (per channel)
        std: Std values (per channel)
        scale_mode: Normalization method used
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, mean=mean, std=std, scale_mode=scale_mode)
    print(f"✓ Saved normalization stats to {output_path}")


def load_normalization_stats(input_path):
    """
    Load saved normalization statistics.

    Args:
        input_path: Path to saved stats

    Returns:
        mean, std, scale_mode
    """
    data = np.load(input_path)
    mean = data['mean']
    std = data['std']
    scale_mode = str(data['scale_mode'])
    print(f"✓ Loaded normalization stats from {input_path}")
    return mean, std, scale_mode


def get_data_summary(X):
    """
    Get quick data summary.

    Args:
        X: Data array

    Returns:
        Summary string
    """
    summary = f"""
Data Summary:
  Shape: {X.shape}
  Dtype: {X.dtype}
  Range: [{X.min():.3f}, {X.max():.3f}]
  Mean: {X.mean():.3f}
  Std: {X.std():.3f}
  Memory: {X.nbytes / 1024 ** 2:.2f} MB
"""
    return summary


# ======================================================================
# TESTING
# ======================================================================

if __name__ == '__main__':


    print("\n" + "=" * 70)
    print("LOAD.PY TEST SUITE - RASTER")
    print("=" * 70)

    dataset_path = "AlmondData/TrainingPatches-4"   # update this
    testset_path = "AlmondData/SmallSet_TestPatches_4"    # update this

    # # --- Test train mode ---
    # print("\n1. Testing train mode...")
    # raster_files, masks = load_raster_data(
    #     dataset_path=dataset_path,
    #     img_size=4,
    #     mode="train",
    #     scale_mode="zscore",
    #     shuffle=True,
    #     max_samples=10,
    # )
    # print(f"raster_files shape : {raster_files.shape}")
    # print(f"masks shape        : {masks.shape}")
    # print(f"raster NaNs        : {np.isnan(raster_files).any()}")
    # print(f"masks NaNs         : {np.isnan(masks).any()}")
    # print(f"raster range       : [{raster_files.min():.3f}, {raster_files.max():.3f}]")
    # print(f"masks range        : [{masks.min():.3f}, {masks.max():.3f}]")

    # --- Test test mode ---
    print("\n2. Testing test mode...")
    raster_test, masks_test = load_raster_data(
        testset_path=testset_path,
        img_size=32,
        mode="test",
        scale_mode="zscore",
    )
    print(f"raster_test shape : {raster_test.shape}")
    print(f"masks_test shape  : {masks_test.shape}")

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70 + "\n")