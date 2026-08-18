import os
import subprocess
import sys


# ============================================================
# MASKING RATIO ABLATION
# ============================================================

MASKING_RATIOS = [0.25, 0.5, 0.85]


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "/content/drive/MyDrive/MAE_grape_pseudoabsences/TrainingPatches_300_32x32"

BASE_OUTPUT_DIR = "/content/drive/MyDrive/MAE_grape_pseudoabsences"


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100
MODEL = "mae_vit_base_patch16"

IMG_SIZE = 32
PATCH_SIZE = 2

BLR = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 40

NUM_WORKERS = 4
SEED = 0


# ============================================================
# RUN EACH MASKING RATIO
# ============================================================

for mask_ratio in MASKING_RATIOS:

    ratio_name = f"{mask_ratio:.2f}"

    # --------------------------------------------------------
    # Separate output directory for each experiment
    # --------------------------------------------------------

    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"outputs_32x32_1600ep_mask{ratio_name}"
    )

    log_dir = os.path.join(
        output_dir,
        "logs"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n")
    print("=" * 80)
    print(f"STARTING MASKING RATIO ABLATION")
    print(f"Masking Ratio: {ratio_name}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # --------------------------------------------------------
    # Build training command
    # --------------------------------------------------------

    command = [
        sys.executable,
        "main_pretrain.py",

        "--batch_size",
        str(BATCH_SIZE),

        "--model",
        MODEL,

        "--img_size",
        str(IMG_SIZE),

        "--patch_size",
        str(PATCH_SIZE),

        "--mask_ratio",
        str(mask_ratio),

        "--epochs",
        str(EPOCHS),

        "--warmup_epochs",
        str(WARMUP_EPOCHS),

        "--blr",
        str(BLR),

        "--weight_decay",
        str(WEIGHT_DECAY),

        "--data_path",
        DATA_PATH,

        "--output_dir",
        output_dir,

        "--log_dir",
        log_dir,

        "--num_workers",
        str(NUM_WORKERS),

        "--seed",
        str(SEED)
    ]

    print("\nRunning:")
    print(" ".join(command))
    print("\n")

    # --------------------------------------------------------
    # Run training
    # --------------------------------------------------------

    result = subprocess.run(command)

    # --------------------------------------------------------
    # Stop if training fails
    # --------------------------------------------------------

    if result.returncode != 0:

        print("\n")
        print("=" * 80)
        print(f"ERROR: MASKING RATIO {ratio_name} FAILED")
        print("=" * 80)

        sys.exit(result.returncode)

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(f"COMPLETED MASKING RATIO: {ratio_name}")
    print(f"Saved to: {output_dir}")
    print("=" * 80)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MASKING RATIO ABLATION COMPLETE")
print("=" * 80)

for mask_ratio in MASKING_RATIOS:

    ratio_name = f"{mask_ratio:.2f}"

    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"outputs_32x32_1600ep_mask{ratio_name}"
    )

    print(
        f"Mask ratio {ratio_name} --> {output_dir}"
    )
