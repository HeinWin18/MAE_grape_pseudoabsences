#  coding:utf-8

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
import model
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from patchGan import (
    masked_disc_loss,
    masked_adv_loss_for_generator,
    masked_pixel_l1_loss, masked_pixel_l2_loss,
)


class DCGAN():
    """
    PatchGAN-based GAN for AnoGAN training.

    This implementation uses:
    - PatchGAN discriminator (outputs spatial logits map)
    - Masked losses (respects b1 presence/absence band)
    - L1 pixel loss + adversarial loss for generator
    """

    def __init__(self, args):
        """
        Initialize DCGAN/PatchGAN model.

        Args:
            args: Namespace with attributes:
                - imgsize: Image size (H=W)
                - channels: Number of channels
                - zdims: Latent dimension
                - epoch: Number of epochs
                - batchsize: Batch size
        """
        self.img_size = args.imgsize
        self.channels = args.channels
        self.z_dim = args.zdims
        self.epochs = args.epoch
        self.batch_size = args.batchsize
        self.lrg = args.lrg
        self.lrd = args.lrd

        # Optimizers
        # Lower learning rate for discriminator to prevent it from overpowering generator
        self.d_opt = Adam(learning_rate=self.lrd, beta_1=0.5)
        self.g_opt = Adam(learning_rate=self.lrg, beta_1=0.5)

        # Create output directories
        self._create_directories()

        # Build models
        print("Building discriminator (PatchGAN)...")
        self.d = model.discriminator_model(self.img_size, self.channels)
        self._save_model_architecture(self.d, 'discriminator.png')

        print("Building generator...")
        self.g = model.generator_model(self.z_dim, self.img_size, self.channels)
        self._save_model_architecture(self.g, 'generator.png')

        print("Building combined model (Generator + Discriminator)...")
        self.d_on_g = model.generator_containg_discriminator(self.g, self.d, self.z_dim)
        self._save_model_architecture(self.d_on_g, 'd_on_g.png')

        # Compile models
        # Note: These compilations are mainly for compatibility
        # Actual training uses custom TensorFlow GradientTape
        self.g.compile(loss='mse', optimizer=self.g_opt)
        self.d_on_g.compile(loss='mse', optimizer=self.g_opt)
        self.d.trainable = True
        self.d.compile(loss='mse', optimizer=self.d_opt)

        print(f"Models initialized successfully!")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Channels: {self.channels}")
        print(f"  Latent dims: {self.z_dim}")

    def _create_directories(self):
        """Create necessary output directories."""
        dirs = ['./result/', './model_images/', './saved_model/']
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _save_model_architecture(self, model, filename):
        """Save model architecture diagram."""
        try:
            plot_model(
                model,
                to_file=f'./model_images/{filename}',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True
            )
            print(f"  [saved] Model architecture -> ./model_images/{filename}")
        except Exception as e:
            print(f"  [warning] Could not save model plot: {e}")

    @tf.function
    def train_step(self, x_real, masks, l1_weight=0.5, use_masking=True, g_steps=1):
        """
        Single training step with PatchGAN.

        Args:
            x_real: (N, H, W, C) real images (last channel is b1 for masking)
            masks: (N, H, W, 1) boolean masks indicating valid pixels
            l1_weight: Weight for L1 pixel loss (default 100.0)
            use_masking: Whether to use b1 masking (default True)

        Returns:
            d_loss: Discriminator loss
            g_loss: Generator total loss
            adv_loss: Adversarial loss component
            l1_loss: L1 pixel loss component
        """

        batch_n = tf.shape(x_real)[0]
        z = tf.random.normal([batch_n, self.z_dim])
        # print(f"Using masking with batch size {batch_n_mask.shape}")


        # Build mask
        if use_masking:
            masks = tf.cast(masks, tf.float32)
            keep = tf.expand_dims(masks, axis=-1) # Ensure shape is (N, H, W, 1)

        else:
            # No masking - all pixels are valid
            keep = tf.ones((batch_n, self.img_size, self.img_size, 1), dtype=tf.float32)

        # -------------------- D update --------------------
        with tf.GradientTape() as tape_d:
            x_fake = self.g(z, training=True)
            # Stop gradient for discriminator update
            d_loss = masked_disc_loss(self.d, x_real, tf.stop_gradient(x_fake), keep)

        d_grads = tape_d.gradient(d_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.d.trainable_variables))

        # -------------------- G update --------------------
        for _ in range(g_steps):
            z = tf.random.normal([batch_n, self.z_dim])  # fresh z each time
            with tf.GradientTape() as tape_g:
                x_fake = self.g(z, training=True)
                adv_loss = masked_adv_loss_for_generator(self.d, x_fake, keep)
                g_loss = adv_loss

            g_grads = tape_g.gradient(g_loss, self.g.trainable_variables)
            self.g_opt.apply_gradients(zip(g_grads, self.g.trainable_variables))

        return d_loss, g_loss

    def train(self, x_real, masks, l1_weight=0.5, use_masking=True, g_steps=1):
        """
        Non-decorated version of train_step for compatibility.
        Called by main.py training loop.

        Args:
            x_real: (N, H, W, C) real images
            masks: foat32 (N, H, W) boolean masks indicating valid pixels (from b1 band)
            l1_weight: Weight for L1 pixel loss
            use_masking: Whether to use b1 masking

        Returns:
            Tuple of (d_loss, g_loss, adv_loss, l1_loss)
        """
        # Convert numpy to tensor if needed
        if isinstance(x_real, np.ndarray):
            x_real = tf.constant(x_real, dtype=tf.float32)
            masks = tf.constant(masks, dtype=tf.float32)

        return self.train_step(x_real, masks, l1_weight=l1_weight, use_masking=use_masking, g_steps=g_steps)

    def generate(self, batch_size, load_weights=True):
        """
        Generate images from random noise.

        Args:
            batch_size: Number of images to generate
            load_weights: Whether to load saved weights first

        Returns:
            Generated images (batch_size, H, W, C)
        """
        if load_weights:
            try:
                self.g.load_weights('./saved_model/generator.weights.h5')
                print("Loaded generator weights from ./saved_model/generator.weights.h5")
            except Exception as e:
                print(f"Warning: Could not load generator weights: {e}")

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        generate_img = self.g.predict(noise, verbose=0)

        return generate_img

    def load_weights(self, generator_path=None, discriminator_path=None):
        """
        Load saved model weights.

        Args:
            generator_path: Path to generator weights (default: ./saved_model/generator.weights.h5)
            discriminator_path: Path to discriminator weights (default: ./saved_model/discriminator.weights.h5)
        """
        if generator_path is None:
            generator_path = './saved_model/generator.weights.h5'
        if discriminator_path is None:
            discriminator_path = './saved_model/discriminator.weights.h5'

        try:
            self.g.load_weights(generator_path)
            print(f"✓ Loaded generator weights from {generator_path}")
        except Exception as e:
            print(f"✗ Could not load generator weights: {e}")

        try:
            self.d.load_weights(discriminator_path)
            print(f"✓ Loaded discriminator weights from {discriminator_path}")
        except Exception as e:
            print(f"✗ Could not load discriminator weights: {e}")

    def save_weights(self, generator_path=None, discriminator_path=None):
        """
        Save model weights.

        Args:
            generator_path: Path to save generator weights
            discriminator_path: Path to save discriminator weights
        """
        if generator_path is None:
            generator_path = './saved_model/generator.weights.h5'
        if discriminator_path is None:
            discriminator_path = './saved_model/discriminator.weights.h5'

        os.makedirs(os.path.dirname(generator_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(discriminator_path) or '.', exist_ok=True)

        self.g.save_weights(generator_path)
        self.d.save_weights(discriminator_path)

        print(f"✓ Saved weights:")
        print(f"  Generator: {generator_path}")
        print(f"  Discriminator: {discriminator_path}")

    def plot_generate_images(self, gen_images):
        """
        Arrange generated images in a grid for visualization.

        Args:
            gen_images: (N, H, W, C) generated images

        Returns:
            Grid image (grid_H, grid_W, C)
        """
        num = gen_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = gen_images.shape[1:4]

        image = np.zeros(
            (height * shape[0], width * shape[1], shape[2]),
            dtype=gen_images.dtype
        )

        for index, img in enumerate(gen_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0],
            j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]

        return image

    def visualize_generation(self, n_samples=9, save_path='./result/generated_samples.png'):
        """
        Generate and visualize samples in a grid.

        Args:
            n_samples: Number of samples to generate (will be rounded to perfect square)
            save_path: Where to save the visualization
        """
        # Make it a perfect square
        grid_size = int(math.sqrt(n_samples))
        n_samples = grid_size * grid_size

        gen_images = self.generate(n_samples)
        grid = self.plot_generate_images(gen_images)

        # Normalize for display
        grid_display = (grid + 1.0) / 2.0  # From [-1, 1] to [0, 1]
        grid_display = np.clip(grid_display, 0, 1)

        # Save
        plt.figure(figsize=(10, 10))
        if grid_display.shape[-1] == 1:
            plt.imshow(grid_display[:, :, 0], cmap='gray')
        else:
            # For multi-channel, show first 3 channels
            plt.imshow(grid_display[:, :, :3])
        plt.axis('off')
        plt.title(f'Generated Samples ({n_samples} images)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved generated samples visualization -> {save_path}")

    def evaluate_reconstruction(self, test_images, n_iterations=100):
        """
        Evaluate generator's ability to reconstruct test images.
        Uses same optimization as AnoGAN but returns reconstruction quality metrics.

        Args:
            test_images: (N, H, W, C) test images
            n_iterations: Number of optimization iterations

        Returns:
            metrics: Dict with reconstruction metrics
        """
        n_samples = test_images.shape[0]

        # Initialize random latent codes
        z = tf.Variable(
            tf.random.normal([n_samples, self.z_dim]),
            trainable=True
        )

        # Optimizer for latent code
        opt = Adam(learning_rate=0.01)

        losses = []

        # Optimize z to reconstruct test images
        for i in range(n_iterations):
            with tf.GradientTape() as tape:
                x_recon = self.g(z, training=False)
                loss = tf.reduce_mean(tf.abs(test_images - x_recon))

            grads = tape.gradient(loss, [z])
            opt.apply_gradients(zip(grads, [z]))
            losses.append(float(loss))

        # Final reconstruction
        x_recon = self.g(z, training=False).numpy()

        # Compute metrics
        mse = np.mean((test_images - x_recon) ** 2)
        mae = np.mean(np.abs(test_images - x_recon))

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'final_loss': losses[-1],
            'losses': losses,
            'reconstructions': x_recon
        }

        return metrics

    def check_mode_collapse(self, n_samples=100, diversity_threshold=0.01):
        """
        Check if generator has suffered mode collapse.

        Args:
            n_samples: Number of samples to generate
            diversity_threshold: Minimum std dev considered diverse

        Returns:
            Dict with mode collapse indicators
        """
        gen_images = self.generate(n_samples, load_weights=False)

        # Compute diversity metrics
        std_per_channel = np.std(gen_images, axis=(0, 1, 2))  # Std across samples and spatial dims
        mean_std = np.mean(std_per_channel)

        # Check if output is too uniform (mode collapse)
        is_collapsed = mean_std < diversity_threshold

        return {
            'is_collapsed': is_collapsed,
            'mean_std': float(mean_std),
            'std_per_channel': std_per_channel.tolist(),
            'diversity_score': float(mean_std)  # Higher = more diverse
        }


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def train_with_validation(dcgan, X_train, X_val=None, epochs=100,
                          save_interval=10, l1_weight=0.5):
    """
    Training loop with optional validation set monitoring.

    Args:
        dcgan: DCGAN instance
        X_train: Training data (N, H, W, C)
        X_val: Validation data (optional)
        epochs: Number of epochs
        save_interval: Save weights every N epochs
        l1_weight: Weight for L1 loss

    Returns:
        history: Dict with training history
    """
    from tqdm import tqdm

    n_samples = X_train.shape[0]
    steps_per_epoch = max(1, n_samples // dcgan.batch_size)

    history = {
        'd_loss': [],
        'g_loss': [],
        'adv_loss': [],
        'l1_loss': [],
        'val_reconstruction': [] if X_val is not None else None
    }

    print(f"Training with {n_samples} samples, {steps_per_epoch} steps per epoch")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Shuffle
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]

        epoch_d, epoch_g, epoch_adv, epoch_l1 = [], [], [], []

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")

        for step in pbar:
            batch_start = step * dcgan.batch_size
            batch_end = min(batch_start + dcgan.batch_size, n_samples)

            if batch_end - batch_start < dcgan.batch_size // 2:
                continue

            x_batch = X_shuffled[batch_start:batch_end]

            # Training step
            d_loss, g_loss= dcgan.train(
                x_batch, l1_weight=l1_weight
            )

            epoch_d.append(float(d_loss))
            epoch_g.append(float(g_loss))

            pbar.set_postfix({
                'D': f'{d_loss:.4f}',
                'G': f'{g_loss:.4f}'

            })

        # Epoch summary
        history['d_loss'].extend(epoch_d)
        history['g_loss'].extend(epoch_g)
        history['adv_loss'].extend(epoch_adv)
        history['l1_loss'].extend(epoch_l1)

        print(f"Epoch {epoch + 1} - D: {np.mean(epoch_d):.4f}, G: {np.mean(epoch_g):.4f}")

        # Validation
        if X_val is not None:
            val_metrics = dcgan.evaluate_reconstruction(X_val[:5], n_iterations=50)
            history['val_reconstruction'].append(val_metrics['mae'])
            print(f"  Validation MAE: {val_metrics['mae']:.4f}")

        # Save weights
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            dcgan.save_weights()

            # Generate samples
            dcgan.visualize_generation(
                n_samples=9,
                save_path=f'./result/generated_epoch_{epoch + 1}.png'
            )

    return history


def plot_training_history(history, save_path='./result/training_history.png'):
    """
    Plot training history curves.

    Args:
        history: Dict from train_with_validation()
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # D loss
    axes[0, 0].plot(history['d_loss'], alpha=0.7)
    axes[0, 0].set_title('Discriminator Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # G loss
    axes[0, 1].plot(history['g_loss'], alpha=0.7, color='orange')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Adv + L1 loss
    axes[1, 0].plot(history['adv_loss'], alpha=0.7, label='Adversarial', color='red')
    axes[1, 0].plot(history['l1_loss'], alpha=0.7, label='L1', color='blue')
    axes[1, 0].set_title('Generator Loss Components')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation (if available)
    if history['val_reconstruction'] is not None:
        axes[1, 1].plot(history['val_reconstruction'], alpha=0.7, color='green')
        axes[1, 1].set_title('Validation Reconstruction Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Validation Data',
                        ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Validation')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✓ Saved training history plot -> {save_path}")