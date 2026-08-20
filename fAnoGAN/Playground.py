# import rasterio
#
# with rasterio.open("presence_1_points_only.tif") as src:
#     print(f"Width: {src.width} pixels")
#     print(f"Height: {src.height} pixels")
#     print(f"Bands: {src.count}")
#     print(f"CRS: {src.crs}")
#     print(f"Data type: {src.dtypes[0]}")
#     print(f"Resolution: {src.res}")

import numpy as np
import tifffile as tiff


# patch = tiff.imread("outputs/similar.tif ")
# print(f"Patch array shape: {patch.shape}")
# print(patch)
#
#
# rgb = patch[..., :3]  # pick 3 bands
# rgb_u8 = np.clip((rgb + 1) * 127.5, 0, 255).astype(np.uint8)
# tiff.imwrite("preview.tif", rgb_u8, photometric="rgb")


import rasterio

# with rasterio.open("train_patches/patch_00009.tif") as src:
#     print(src.meta)          # summary
#     print(src.tags())        # global tags
#     for i in range(1, src.count + 1):
#         print(f"Band {i} stats:", src.statistics(i))


import numpy as np, tifffile as tiff, matplotlib.pyplot as plt

def _to_disp(arr):
    a = np.nan_to_num(arr.astype(np.float32))
    if a.min() >= -1.1 and a.max() <= 1.1:   # typical GAN range
        a = (a + 1.0) / 2.0
    else:
        a = (a - a.min()) / (np.ptp(a) + 1e-8)
    return np.clip(a, 0, 1)

def _first3(a):  # pick display channels
    return a[...,0] if a.shape[-1] == 1 else a[...,:3]

def illustrate(original_hwC, similar_hwC, residual, base="outputs/"):
    # Save TIFFs (float32 preserves range)
    tiff.imwrite(f"{base}original.tif", original_hwC.astype(np.float32))
    tiff.imwrite(f"{base}similar.tif",  similar_hwC.astype(np.float32))
    tiff.imwrite(f"{base}residual.tif", residual.astype(np.float32))

    # Print quick stats
    for name, x in [("original", original_hwC), ("similar", similar_hwC), ("residual", residual)]:
        print(f"{name}: shape={x.shape} dtype={x.dtype} min={x.min():.4f} max={x.max():.4f}")

    # Visualize
    o = _to_disp(_first3(original_hwC))
    s = _to_disp(_first3(similar_hwC))
    r = _to_disp(_first3(residual))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(o, cmap="gray" if o.ndim==2 else None); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Similar");  plt.imshow(s, cmap="gray" if s.ndim==2 else None); plt.axis("off")
    plt.subplot(1,3,3); plt.title("Residual"); plt.imshow(r, cmap="gray" if r.ndim==2 else None); plt.axis("off")
    plt.tight_layout(); plt.show()


# ---------------------------- Heat Map -------------------------
#
# def load_stack(path):
#     with rasterio.open(path) as src:
#         return src.read().astype(np.float32)  # (bands, H, W)
#
# # ---- file paths ----
# path1 = "outputs/original.tif"
# path2 = "outputs/similar.tif"
#
# # ---- load ----
# A = load_stack(path1)
# B = load_stack(path2)
#
# assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"
#
# # ---- compute difference heatmap ----
# diff = np.abs(A - B)        # (bands, H, W)
# heatmap = np.mean(diff, 0)  # average over bands
#
# # ---- show ----
# plt.imshow(heatmap, cmap="hot")
# plt.colorbar(label="Mean abs diff")
# plt.title("Heatmap of differences across all bands")
# plt.axis("off")
# plt.show()

# ----------------------------------------------------------------

# import rasterio
#
# # Open the raster file
# with rasterio.open('stacked_bioclim.tif') as src:
#     width = src.width
#     height = src.height
#     print(f"Raster dimensions: {width} columns x {height} rows")


# --------------------------- showing npy file ---------------------------
# import rasterio
# import numpy as np
#
# with rasterio.open("AlmondData/TrainingPatches-8/patch_00000.tif") as src:
#     img = src.read()  # shape: (bands, H, W)
#
# # Convert to (H, W, C)
# img = np.transpose(img, (1, 2, 0))
#
# # First pixel (row=0, col=0), all bands
# first_pixel = img[0,0,:]
#
# print("First pixel channel values:")
# print(first_pixel)
# print("type:", first_pixel.dtype)


