#this is model.py
# coding: utf-8
import numpy as np
import os
from datetime import datetime
import cv2
import math
import tensorflow as tf

from tensorflow.keras import layers, initializers, backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten,
    Conv2DTranspose, LeakyReLU, Activation, BatchNormalization, Lambda
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model, Progbar
from patchGan import patchgan_discriminator_model
import mlflow

def encoder_model(imgsize, channels, z_dim):
    """
    Encoder: maps real patch → latent vector z
    Mirror of Generator architecture (reversed)
    """
    inputs = Input((imgsize, imgsize, channels), name='encoder_input')

    x = Conv2D(64, (3, 3), padding='same', name='enc_conv1')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', name='enc_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', name='enc_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    outputs = Dense(z_dim, name='enc_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Encoder')
    model.summary()
    return model

#Orginal loss
def train_encoder(args, g, d, X_train, masks, epochs=500, save_path='./saved_model/'):
    """
    Train encoder after GAN is already trained.
    Generator and Discriminator weights are FROZEN.
    Only Encoder weights update.
    """
    # Build encoder
    encoder = encoder_model(args.imgsize, args.channels, args.zdims)

    # Freeze G and D
    g.trainable = False
    d.trainable = False

    # Feature extractor from Discriminator (same as anomaly detection)
    feat_model = feature_extractor(args, d)

    # optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Training encoder for {epochs} epochs...")
    
    # SANITY CHECK — overfit 1 batch
    print("=== SANITY CHECK: overfitting 1 batch ===")
    x_single = tf.constant(X_train[:args.batchsize], dtype=tf.float32)
    mask_single = tf.constant(masks[:args.batchsize], dtype=tf.float32)
    keep_single = tf.expand_dims(mask_single, axis=-1)

    for i in range(200):
        with tf.GradientTape() as tape:
            z_pred = encoder(x_single, training=True)
            x_recon = g(z_pred, training=False)
            img_diff = tf.abs(x_single - x_recon) * keep_single
            img_loss = tf.reduce_sum(img_diff) / (tf.reduce_sum(keep_single) + 1e-7)
        grads = tape.gradient(img_loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
        if i % 20 == 0:
            print(f"  Step {i}: img_loss = {float(img_loss):.4f}")

    print("=== END SANITY CHECK ===")

    for epoch in range(epochs):
        # shuffle
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        masks_shuffled = masks[idx]

        epoch_losses = []
        epoch_img_losses = []    # ADD
        epoch_feat_losses = []  

        steps = max(1, len(X_train) // args.batchsize)
        for step in range(steps):
            x_batch = X_shuffled[step * args.batchsize:(step + 1) * args.batchsize]
            mask_batch = masks_shuffled[step * args.batchsize:(step + 1) * args.batchsize]

            x_batch = tf.constant(x_batch, dtype=tf.float32)
            mask_batch = tf.constant(mask_batch, dtype=tf.float32)
            keep = tf.expand_dims(mask_batch, axis=-1)  # (N, H, W, 1)

            with tf.GradientTape() as tape:
                # Encoder forward pass
                z_pred = encoder(x_batch, training=True)

                # Generator reconstructs from predicted z
                x_recon = g(z_pred, training=False)

                # Feature extraction from Discriminator
                d_real = feat_model(x_batch,  training=False)
                d_recon = feat_model(x_recon, training=False)

                #x_batch_clipped = tf.clip_by_value(x_batch, -1.0, 1.0)

                # Image loss — masked to valid pixels only
                img_diff = tf.abs(x_batch - x_recon) * keep
                img_loss = tf.reduce_sum(img_diff) / (tf.reduce_sum(keep) + 1e-7)

                # Feature loss -- applying masking there. 
                #keep # shape: (N, H, W, 1) = (64, 4, 4, 1)
                # d_real   # shape: (N, H', W', C_feat)
                # d_recon  # shape: (N, H', W', C_feat) -- feature maps
                # feat_mask -- resized mask to match feature map size
                # feat_diff — where real and reconstructed features disagree, on valid pixels only cuz * feat_mask — zeros out the difference at invalid pixel locations (NaN/background)
                feat_mask = tf.image.resize(keep, [tf.shape(d_real)[1], tf.shape(d_real)[2]], method='nearest')
                feat_diff = tf.abs(d_real - d_recon) * feat_mask #Result: only valid pixels contribute to the feature loss
                feat_loss = tf.reduce_sum(feat_diff) / (tf.reduce_sum(feat_mask) + 1e-7)

                total_loss = 0.5 * img_loss + 0.1 * feat_loss

            grads = tape.gradient(total_loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_losses.append(float(total_loss))
            epoch_img_losses.append(float(img_loss))
            epoch_feat_losses.append(float(feat_loss))
        avg_loss = np.mean(epoch_losses)
        avg_img = np.mean(epoch_img_losses)
        avg_feat = np.mean(epoch_feat_losses)
        print(f"Epoch {epoch + 1}/{epochs} — Total: {avg_loss:.4f} | Img: {avg_img:.4f} | Feat: {avg_feat:.4f}")

        # Save periodically
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            os.makedirs(save_path, exist_ok=True)
            encoder.save(f'{save_path}/encoder_{timestamp}.h5')
            print(f"[saved] encoder checkpoint → {save_path}/encoder_{timestamp}.h5")

    return encoder


#Testing Loss
def train_encoder_new_loss(args, g, d, X_train, masks, epochs=500, save_path='./saved_model/'):
    encoder = encoder_model(args.imgsize, args.channels, args.zdims)

    g.trainable = False
    d.trainable = False

    feat_model = feature_extractor(args, d)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Training encoder for {epochs} epochs...")

    print("=== SANITY CHECK: overfitting 1 batch ===")
    x_single = tf.constant(X_train[:args.batchsize], dtype=tf.float32)
    mask_single = tf.constant(masks[:args.batchsize], dtype=tf.float32)
    keep_single = tf.expand_dims(mask_single, axis=-1)

    for i in range(200):
        with tf.GradientTape() as tape:
            z_pred = encoder(x_single, training=True)
            x_recon = g(z_pred, training=False)
            img_diff = tf.abs(x_single - x_recon) * keep_single
            img_loss = tf.reduce_sum(img_diff) / (tf.reduce_sum(keep_single) + 1e-7)
        grads = tape.gradient(img_loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
        if i % 20 == 0:
            print(f"  Step {i}: img_loss = {float(img_loss):.4f}")

    print("=== END SANITY CHECK ===")

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        masks_shuffled = masks[idx]

        epoch_losses, epoch_img_losses, epoch_feat_losses = [], [], []

        steps = max(1, len(X_train) // args.batchsize)
        for step in range(steps):
            x_batch = X_shuffled[step * args.batchsize:(step + 1) * args.batchsize]
            mask_batch = masks_shuffled[step * args.batchsize:(step + 1) * args.batchsize]

            x_batch = tf.constant(x_batch, dtype=tf.float32)
            mask_batch = tf.constant(mask_batch, dtype=tf.float32)
            keep = tf.expand_dims(mask_batch, axis=-1)

            with tf.GradientTape() as tape:
                z_pred = encoder(x_batch, training=True)
                x_recon = g(z_pred, training=False)
                d_real = feat_model(x_batch, training=False)
                d_recon = feat_model(x_recon, training=False)

                # CHANGED: tf.square (MSE) instead of tf.abs (MAE)
                img_diff = tf.square(x_batch - x_recon) * keep
                img_loss = tf.reduce_sum(img_diff) / (tf.reduce_sum(keep) + 1e-7)

                feat_mask = tf.image.resize(keep, [tf.shape(d_real)[1], tf.shape(d_real)[2]], method='nearest')
                feat_diff = tf.square(d_real - d_recon) * feat_mask  # CHANGED: tf.square
                feat_loss = tf.reduce_sum(feat_diff) / (tf.reduce_sum(feat_mask) + 1e-7)

                total_loss = img_loss + 1.0 * feat_loss  # CHANGED: kappa=1.0, no 0.5

            grads = tape.gradient(total_loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_losses.append(float(total_loss))
            epoch_img_losses.append(float(img_loss))
            epoch_feat_losses.append(float(feat_loss))

        avg_loss = np.mean(epoch_losses)
        avg_img = np.mean(epoch_img_losses)
        avg_feat = np.mean(epoch_feat_losses)
        print(f"Epoch {epoch + 1}/{epochs} — Total: {avg_loss:.4f} | Img: {avg_img:.4f} | Feat: {avg_feat:.4f}")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            os.makedirs(save_path, exist_ok=True)
            encoder.save(f'{save_path}/encoder_{timestamp}.h5')
            print(f"[saved] encoder → {save_path}/encoder_{timestamp}.h5")

    return encoder




# ======================================================================
# GENERATOR MODEL
# ======================================================================

def generator_model(z_dim, imgsize, channels, base_filters=64):
    """
    Generator model with dynamic architecture based on image size.

    Architecture:
        - Starts from (imgsize/4 × imgsize/4 × 128) spatial resolution
        - Two 2× upsampling layers to reach target imgsize
        - Works for imgsize divisible by 4 (4, 8, 12, 16, 32, etc.)

    Args:
        z_dim: Latent dimension (e.g., 100)
        imgsize: Target image size H=W (must be divisible by 4)
        channels: Number of output channels
        base_filters: Base number of filters (default 64)

    Returns:
        Keras Model

    Raises:
        ValueError: If imgsize is not divisible by 4
    """
    # Validate imgsize
    if imgsize % 4 != 0:
        raise ValueError(
            f"imgsize must be divisible by 4 for current architecture. "
            f"Got {imgsize}. Try: 4, 8, 12, 16, 20, 24, 28, 32, etc."
        )

    # Starting spatial size (will be upsampled 2x twice = 4x total)
    # start_size = int(imgsize / 4)
    start_size = max(1, int(imgsize / 4))

    # Scale filters based on image size for better capacity
    initial_filters = 128
    mid_filters = base_filters

    # Input: latent vector
    inputs = Input((z_dim,), name='latent_input')

    # Dense layer to create initial spatial structure
    x = Dense(initial_filters * start_size * start_size, name='dense_projection')(inputs)
    x = BatchNormalization(name='bn_dense')(x)
    x = LeakyReLU(0.2, name='lrelu_dense')(x)

    # Reshape to spatial: (start_size, start_size, 128)
    x = Reshape((start_size, start_size, initial_filters), name='reshape_spatial')(x)

    # First upsampling: (start_size, start_size, 128) → (2*start_size, 2*start_size, 64)
    x = Conv2DTranspose(
        mid_filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        name='upsample_1'
    )(x)
    x = BatchNormalization(name='bn_up1')(x)
    x = LeakyReLU(0.2, name='lrelu_up1')(x)

    # Refinement conv
    x = Conv2D(mid_filters, (3, 3), padding='same', name='conv_refine_1')(x)
    x = BatchNormalization(name='bn_refine_1')(x)
    x = Activation('relu', name='relu_refine_1')(x)

    # Second upsampling: (2*start_size, 2*start_size, 64) → (imgsize, imgsize, 64)
    x = Conv2DTranspose(
        mid_filters,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding='same',
        name='upsample_2'
    )(x)
    x = BatchNormalization(name='bn_up2')(x)
    x = LeakyReLU(0.2, name='lrelu_up2')(x)

    # Final output layer: (imgsize, imgsize, channels)
    outputs = Conv2D(
        channels,
        kernel_size=(5, 5),
        padding='same',
        activation='tanh',
        name='output_layer'
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')

    # Print summary
    model.summary()

    # Save architecture diagram
    try:
        plot_model(
            model,
            to_file='./model_images/Generator.png',
            show_shapes=True,
            show_layer_names=True
        )
    except Exception as e:
        print(f"Warning: Could not save generator architecture plot: {e}")

    return model


# ======================================================================
# DISCRIMINATOR MODEL (wrapper)
# ======================================================================

def discriminator_model(img_size, channels):
    """
    Discriminator model wrapper - uses PatchGAN implementation.

    Args:
        img_size: Image size H=W
        channels: Number of input channels

    Returns:
        PatchGAN discriminator model
    """
    return patchgan_discriminator_model(img_size, channels)


# ======================================================================
# COMBINED GAN MODEL
# ======================================================================

def generator_containg_discriminator(g, d, z_dim):
    """
    Combined Generator + Discriminator model for adversarial training.

    Args:
        g: Generator model
        d: Discriminator model
        z_dim: Latent dimension

    Returns:
        Combined model (z → G(z) → D(G(z)))
    """
    # Freeze discriminator when training generator through combined model
    d.trainable = False

    ganInput = Input(shape=(z_dim,), name='gan_latent_input')
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput, name='GAN_Combined')

    return gan


# ======================================================================
# FEATURE EXTRACTOR (FIXED)
# ======================================================================

def feature_extractor(args, d=None):
    """
    Extract intermediate features from PatchGAN discriminator.

    CRITICAL FIX: The layer index has been corrected for PatchGAN architecture.

    PatchGAN structure:
        0: InputLayer
        1: Conv2D (64 filters)
        2: LeakyReLU
        3: Conv2D (64 filters)
        4: BatchNormalization
        5: LeakyReLU ← Good feature extraction point
        6: Conv2D (64 filters)
        7: BatchNormalization
        8: LeakyReLU
        9: Conv2D (64 filters)
        10: LeakyReLU
        11: Conv2D (1 filter, final logits)

    Args:
        args: Namespace with imgsize and channels
        d: PatchGAN discriminator (if None, creates new one)

    Returns:
        Model that outputs intermediate features (spatial feature map)
    """
    if d is None:
        print("No discriminator received, creating new PatchGAN discriminator...")
        d = patchgan_discriminator_model(args.imgsize, args.channels)

    # FIXED: Use appropriate layer for PatchGAN
    # Layer 5 is after 2nd Conv+BN+LeakyReLU (good mid-level features)
    # Alternative: layer 8 for higher-level features

    # You can adjust this based on your needs:
    # - Earlier layers (2, 5): More local, texture-like features
    # - Later layers (8, 10): More semantic, higher-level features

    feature_layer_idx = 5  # After 2nd Conv2D+BN+LeakyReLU

    print(f"Creating feature extractor from layer {feature_layer_idx}: {d.layers[feature_layer_idx].name}")

    intermediate_model = Model(
        inputs=d.input,
        outputs=d.layers[feature_layer_idx].output,
        name='Feature_Extractor'
    )

    # Compile (mainly for compatibility)
    intermediate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # Save architecture
    try:
        plot_model(
            intermediate_model,
            to_file='./model_images/feature_extractor.png',
            show_shapes=True
        )
    except Exception as e:
        print(f"Warning: Could not save feature extractor plot: {e}")

    return intermediate_model


# ======================================================================
# LOSS FUNCTIONS
# ======================================================================

def sum_of_residual(y_true, y_pred):
    """
    Residual loss function for anomaly detection.
    Uses mean absolute error across all spatial locations and channels.

    Args:
        y_true: Ground truth (H, W, C) or (N, H, W, C)
        y_pred: Prediction (same shape)

    Returns:
        Scalar loss
    """
    return K.mean(K.abs(y_true - y_pred))


def L2_residual(y_true, y_pred):
    """
    L2 residual loss function for anomaly detection.
    Uses mean squared error across all spatial locations and channels.

    Args:
        y_true: Ground truth (H, W, C) or (N, H, W, C)
        y_pred: Prediction (same shape)

    Returns:
        Scalar loss
    """
    return K.mean(K.square(y_true - y_pred))


def spatial_residual(y_true, y_pred):
    """
    Compute spatial residual map (not aggregated).
    Useful for visualization of pixel-wise anomaly scores.

    Args:
        y_true: Ground truth
        y_pred: Prediction

    Returns:
        Residual map (same shape as input)
    """
    return K.abs(y_true - y_pred)


# ======================================================================
# ANOMALY DETECTOR MODEL
# ======================================================================

def anomaly_detector(args, g=None, d=None):
    """
    Build the AnoGAN anomaly detection model.

    This model optimizes a latent code z to minimize:
        Loss = α * |x - G(z)| + β * |f(x) - f(G(z))|
    where:
        - x is the query image
        - G(z) is the generated image
        - f() extracts discriminator features
        - α, β are loss weights (default 0.95, 0.05)

    Args:
        args: Namespace with zdims, imgsize, channels
        g: Generator model (optional)
        d: Discriminator model (optional)

    Returns:
        Keras Model for anomaly detection
    """
    # Load or create generator
    if g is None or d is None:
        print("No generator is received, creating new generator...")
        g = generator_model(args.zdims, args.imgsize, args.channels)
        # Optionally load weights here if needed

    # Create feature extractor from discriminator
    intermediate_model = feature_extractor(args, d)
    intermediate_model.trainable = False

    # Extract generator's computation (skip input layer)
    # This allows us to add a trainable input transformation
    g_functional = Model(
        inputs=g.layers[1].input,
        outputs=g.layers[-1].output,
        name='Generator_Functional'
    )
    g_functional.trainable = False

    # Trainable latent code input layer
    # This allows optimization of z during anomaly detection
    aInput = Input(shape=(args.zdims,), name='anomaly_latent_input')
    # gInput = Dense(args.zdims, trainable=True, name='latent_mapper')(aInput)
    # gInput = Activation('sigmoid', name='latent_activation')(gInput)

    # ------------------------ trying diff input with tanh and scaling ------------------------
    gInput = Dense(args.zdims, trainable=True, name='latent_mapper')(aInput)
    gInput = Activation('tanh', name='tanh_activation')(gInput)
    gInput = Lambda(lambda x: x * 3, name='scale_activation')(gInput)

    # Forward pass through generator and feature extractor
    G_out = g_functional(gInput)  # Generated image
    D_out = intermediate_model(G_out)  # Discriminator features

    # Create model with two outputs: [generated_image, discriminator_features]
    model = Model(
        inputs=aInput,
        outputs=[G_out, D_out],
        name='Anomaly_Detector'
    )
    # Compile with dual losses
    # Loss 1: Reconstruction (pixel-wise similarity)
    # Loss 2: Feature matching (discriminator feature similarity)

    lr = 0.00002
    model.compile(
        loss=[L2_residual, sum_of_residual],
        loss_weights=[0.5, 0.5],
        optimizer=Adam(learning_rate=lr)
    )
    # track lr in mlflow
    mlflow.log_params({
        'learning_rate': lr,
    })

    # Save architecture
    try:
        plot_model(
            model,
            to_file='./model_images/anomaly_detector.png',
            show_shapes=True
        )
    except Exception as e:
        print(f"Warning: Could not save anomaly detector plot: {e}")

    return model


# ======================================================================
# ANOMALY SCORE COMPUTATION
# ======================================================================

# def compute_anomaly_score(args, model, x, iterations=500, d=None, verbose=0):
#     """
#     Compute anomaly score by optimizing latent code to reconstruct input.
#
#     Process:
#         1. Initialize random latent code z
#         2. Optimize z to minimize reconstruction loss
#         3. Return final loss as anomaly score
#
#     Args:
#         args: Namespace with zdims
#         model: Anomaly detector model
#         x: Query image (1, H, W, C)
#         iterations: Number of optimization iterations (default 500)
#         d: Discriminator (optional, for feature extraction)
#         verbose: Verbosity level (0=silent, 1=progress bar, 2=per-iteration)
#
#     Returns:
#         loss: Scalar anomaly score (lower = more normal)
#         similar_data: Reconstructed image (1, H, W, C)
#     """
#     # Initialize random latent code
#     z = np.random.uniform(0, 1, size=(1, args.zdims))
#
#     # Extract discriminator features of query image
#     intermediate_model = feature_extractor(args, d)
#     d_x = intermediate_model.predict(x, verbose=0)
#
#     # Optimize z to reconstruct x
#     history = model.fit(
#         z,
#         [x, d_x],  # Target: [query image, query features]
#         batch_size=1,
#         epochs=iterations,
#         verbose=verbose
#     )
#
#     # Generate final reconstruction
#     similar_data, _ = model.predict(z, verbose=0)
#
#     # Extract final loss as anomaly score
#     final_loss = history.history['loss'][-1]
#
#     return final_loss, similar_data

def compute_anomaly_score(args, model, x, mask, iterations=500, d=None, verbose=1, use_masking=True):
    """
    Compute anomaly score by optimizing latent code to reconstruct input.

    Process:
        1. Initialize random latent code z
        2. Optimize z to minimize reconstruction loss (only on valid pixels)
        3. Return final loss as anomaly score

    Args:
        args: Namespace with zdims
        model: Anomaly detector model
        x: Query image (1, H, W, C)
        iterations: Number of optimization iterations (default 500)
        d: Discriminator (optional, for feature extraction)
        verbose: Verbosity level (0=silent, 1=progress bar, 2=per-iteration)
        use_masking: Whether to mask invalid pixels (default True)

    Returns:
        loss: Scalar anomaly score (lower = more normal, computed only on valid pixels)
        similar_data: Reconstructed image (1, H, W, C)
    """
    # Initialize random latent code
    z = np.random.normal(0, 1, size=(1, args.zdims))

    # Extract discriminator features of query image
    intermediate_model = feature_extractor(args, d)
    d_x = intermediate_model.predict(x, verbose=0)

    # Create validity mask if using masking
    if use_masking:

        mask = mask.astype(np.int32)
        n_valid = np.sum(mask)
        print(f"Valid pixels in mask: {n_valid} / {mask.size} ({100 * n_valid / mask.size:.2f}%)")

        if n_valid == 0:
            print("⚠️ Warning: No valid pixels found! Disabling masking.")
            use_masking = False
            # mask = np.ones(x.shape[1:-1], dtype=np.int32)  # All pixels valid
            mask = np.ones((x.shape[0],), dtype=np.int32)
    # Optimize z to reconstruct x
    history = model.fit(
        z,
        [x, d_x],  # Target: [query image, query features]
        batch_size=1,
        epochs=iterations,
        verbose=verbose,
        sample_weight=[mask, mask]
    )



    # Generate final reconstruction
    similar_data, _ = model.predict(z, verbose=0)

    # Extract final loss as anomaly score
    final_loss = history.history['loss'][-1]


    # Compute masked loss if requested
    if use_masking:
        # Recompute loss only on valid pixels
        x_np = x[0]  # (H, W, C)
        similar_np = tf.squeeze(similar_data, axis=0) * mask  # (H, W, C)
        # Reconstruction component (95%)
        residual_map = np.mean(np.abs(x_np - similar_np), axis=-1)  # (H, W)
        masked_residual = np.mean(residual_map * mask)  # Average only over valid pixels

        # Feature component (5%)
        d_similar = intermediate_model.predict(similar_data, verbose=0)
        feature_diff = np.abs(d_x - d_similar)

        # Mask feature difference
        feature_map = np.mean(feature_diff[0], axis=-1)  # Average over channels

        # Create spatial mask for features (same size as feature map)
        if feature_map.shape == mask.shape:
            masked_feature = np.mean(feature_map[mask])
        else:
            # If feature map is different size, just use mean
            masked_feature = np.mean(feature_diff)

        # Combined masked loss (same weights as training: 0.95, 0.05)
        masked_loss = 0.5 * masked_residual + 0.5 * masked_feature

        if verbose > 0:
            print(f"\nLoss comparison:")
            print(f"  Original (all pixels): {final_loss:.6f}")
            print(f"  Masked (valid only):   {masked_loss:.6f}")
            print(f"  Valid pixels: {n_valid} / {mask.size}")

        return masked_loss, similar_data

    return final_loss, similar_data


def compute_anomaly_score_with_components(args, model, x, iterations=500, d=None, verbose=0):
    """
    Enhanced version that returns loss components separately.

    Returns:
        total_loss: Overall anomaly score
        reconstruction_loss: Pixel-wise reconstruction error
        feature_loss: Feature matching error
        similar_data: Reconstructed image
        history: Full training history
    """
    z = np.random.uniform(0, 1, size=(1, args.zdims))

    intermediate_model = feature_extractor(args, d)
    d_x = intermediate_model.predict(x, verbose=0)

    # Train
    history = model.fit(
        z,
        [x, d_x],
        batch_size=1,
        epochs=iterations,
        verbose=verbose
    )

    # Generate reconstruction
    similar_data, d_similar = model.predict(z, verbose=0)

    # Extract loss components
    total_loss = history.history['loss'][-1]

    # If history contains component losses (depends on Keras version)
    if 'output_1_loss' in history.history:
        reconstruction_loss = history.history['output_1_loss'][-1]
        feature_loss = history.history['output_2_loss'][-1]
    else:
        # Compute manually
        reconstruction_loss = np.mean(np.abs(x - similar_data))
        feature_loss = np.mean(np.abs(d_x - d_similar))

    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'feature_loss': feature_loss,
        'similar_data': similar_data,
        'history': history.history
    }


# ======================================================================
# BATCH ANOMALY SCORING
# ======================================================================

def compute_batch_anomaly_scores(args, model, X, iterations=100, d=None, verbose=0):
    """
    Compute anomaly scores for multiple images efficiently.

    Args:
        args: Namespace with zdims
        model: Anomaly detector model
        X: Batch of images (N, H, W, C)
        iterations: Optimization iterations per image
        d: Discriminator (optional)
        verbose: Verbosity level

    Returns:
        scores: List of anomaly scores
        reconstructions: Array of reconstructed images (N, H, W, C)
    """
    n_samples = X.shape[0]
    scores = []
    reconstructions = []

    print(f"Computing anomaly scores for {n_samples} images...")

    for i in range(n_samples):
        if verbose > 0:
            print(f"Processing image {i + 1}/{n_samples}...", end='\r')

        x_single = X[i:i + 1]  # Keep batch dimension
        score, recon = compute_anomaly_score(
            args, model, x_single,
            iterations=iterations,
            d=d,
            verbose=0
        )

        scores.append(score)
        reconstructions.append(recon[0])

    if verbose > 0:
        print()  # New line after progress

    reconstructions = np.stack(reconstructions, axis=0)

    return scores, reconstructions


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def get_model_summary_info(model):
    """
    Get detailed information about a model's architecture.

    Args:
        model: Keras model

    Returns:
        Dict with model info
    """
    info = {
        'name': model.name,
        'n_layers': len(model.layers),
        'n_params': model.count_params(),
        'trainable_params': sum([K.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([K.count_params(w) for w in model.non_trainable_weights]),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
    }

    return info


def print_model_info(model, name=None):
    """
    Print readable model information.

    Args:
        model: Keras model
        name: Optional name to display
    """
    info = get_model_summary_info(model)

    print("\n" + "=" * 60)
    print(f"MODEL INFO: {name or info['name']}")
    print("=" * 60)
    print(f"Total parameters: {info['n_params']:,}")
    print(f"  Trainable: {info['trainable_params']:,}")
    print(f"  Non-trainable: {info['non_trainable_params']:,}")
    print(f"Input shape: {info['input_shape']}")
    print(f"Output shape: {info['output_shape']}")
    print(f"Number of layers: {info['n_layers']}")
    print("=" * 60 + "\n")


def verify_patchgan_layer_indices(img_size=8, channels=20):
    """
    Helper function to verify PatchGAN layer indices.
    Run this to see the architecture and choose the best feature extraction layer.

    Args:
        img_size: Image size for PatchGAN
        channels: Number of channels
    """
    print("\n" + "=" * 70)
    print("PATCHGAN DISCRIMINATOR LAYER STRUCTURE")
    print("=" * 70)

    d = patchgan_discriminator_model(img_size, channels)

    print(f"\nTotal layers: {len(d.layers)}\n")

    for i, layer in enumerate(d.layers):
        try:
            output_shape = layer.output_shape
        except:
            output_shape = "N/A"

        print(f"Layer {i:2d}: {layer.name:25s} → {output_shape}")

    print("\n" + "=" * 70)
    print("RECOMMENDED FEATURE EXTRACTION LAYERS:")
    print("  - Layer 5: After 2nd Conv+BN+LeakyReLU (mid-level features)")
    print("  - Layer 8: After 3rd Conv+BN+LeakyReLU (higher-level features)")
    print("=" * 70 + "\n")


# ======================================================================
# TESTING/DEBUGGING FUNCTIONS
# ======================================================================

if __name__ == '__main__':
    """
    Test the model building functions.
    Run this file directly to verify everything works.
    """
    import argparse

    print("\n" + "=" * 70)
    print("MODEL.PY TEST SUITE")
    print("=" * 70 + "\n")

    # Test parameters
    args = argparse.Namespace(
        imgsize=8,
        channels=20,
        zdims=100
    )

    # Test 1: Generator
    print("Test 1: Building Generator...")
    try:
        g = generator_model(args.zdims, args.imgsize, args.channels)
        print_model_info(g, "Generator")
        print("✓ Generator build successful!\n")
    except Exception as e:
        print(f"✗ Generator build failed: {e}\n")

    # Test 2: Discriminator
    print("Test 2: Building Discriminator...")
    try:
        d = discriminator_model(args.imgsize, args.channels)
        print_model_info(d, "Discriminator (PatchGAN)")
        print("✓ Discriminator build successful!\n")
    except Exception as e:
        print(f"✗ Discriminator build failed: {e}\n")

    # Test 3: Feature Extractor
    print("Test 3: Building Feature Extractor...")
    try:
        verify_patchgan_layer_indices(args.imgsize, args.channels)
        fe = feature_extractor(args, d)
        print_model_info(fe, "Feature Extractor")
        print("✓ Feature extractor build successful!\n")
    except Exception as e:
        print(f"✗ Feature extractor build failed: {e}\n")

    # Test 4: Anomaly Detector
    print("Test 4: Building Anomaly Detector...")
    try:
        ad = anomaly_detector(args, g, d)
        print_model_info(ad, "Anomaly Detector")
        print("✓ Anomaly detector build successful!\n")
    except Exception as e:
        print(f"✗ Anomaly detector build failed: {e}\n")

    # Test 5: Forward pass
    print("Test 5: Testing forward pass...")
    try:
        # Generate random input
        z_test = np.random.normal(0, 1, (2, args.zdims))
        x_test = np.random.randn(2, args.imgsize, args.imgsize, args.channels).astype(np.float32)

        # Test generator
        gen_out = g.predict(z_test, verbose=0)
        print(f"  Generator output shape: {gen_out.shape}")

        # Test discriminator
        disc_out = d.predict(x_test, verbose=0)
        print(f"  Discriminator output shape: {disc_out.shape}")

        # Test feature extractor
        feat_out = fe.predict(x_test, verbose=0)
        print(f"  Feature extractor output shape: {feat_out.shape}")

        print("✓ All forward passes successful!\n")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}\n")

    print("=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70 + "\n")
