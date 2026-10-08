# ASSIGNMENT 10
# Comparing GAN and VAE Using Experimental Results

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# IMPORTANT:
# This code assumes that your trained models are already available
# as:
#   generator -> GAN generator from Assignment 4
#   decoder   -> VAE decoder from Assignment 8
#
# If you saved the models as files, load them first, for example:
#
# from tensorflow.keras.models import load_model
# generator = load_model("gan_generator.h5")
# decoder = load_model("vae_decoder.h5")
# ============================================================


# ============================================================
# 1. GENERATE 8 IMAGES FROM GAN
# ============================================================

latent_dim = 100

noise = np.random.normal(0, 1, (8, latent_dim))

gan_images = generator.predict(noise)


# ============================================================
# 2. GENERATE 8 IMAGES FROM VAE
# ============================================================

vae_latent_dim = 2

z = np.random.normal(0, 1, (8, vae_latent_dim))

vae_images = decoder.predict(z)


# ============================================================
# 3. FUNCTION TO DISPLAY GENERATED IMAGES
# ============================================================

def display_images(images, title):

    plt.figure(figsize=(12, 6))

    for i in range(8):

        plt.subplot(2, 4, i + 1)

        image = images[i]

        if len(image.shape) == 3 and image.shape[-1] == 1:
            plt.imshow(image.squeeze(), cmap="gray")
        else:
            plt.imshow(image)

        plt.axis("off")
        plt.title("Image " + str(i + 1))

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. DISPLAY GAN IMAGES
# ============================================================

display_images(
    gan_images,
    "GAN Generated Images - Assignment 4"
)


# ============================================================
# 5. DISPLAY VAE IMAGES
# ============================================================

display_images(
    vae_images,
    "VAE Generated Images - Assignment 8"
)


# ============================================================
# 6. DISPLAY GAN AND VAE TOGETHER
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(8):

    # GAN image
    plt.subplot(2, 8, i + 1)

    image = gan_images[i]

    if len(image.shape) == 3 and image.shape[-1] == 1:
        plt.imshow(image.squeeze(), cmap="gray")
    else:
        plt.imshow(image)

    plt.axis("off")

    # VAE image
    plt.subplot(2, 8, i + 9)

    image = vae_images[i]

    if len(image.shape) == 3 and image.shape[-1] == 1:
        plt.imshow(image.squeeze(), cmap="gray")
    else:
        plt.imshow(image)

    plt.axis("off")


plt.suptitle("GAN vs VAE Generated Images", fontsize=16)

plt.figtext(
    0.5,
    0.02,
    "Top Row: GAN    |    Bottom Row: VAE",
    ha="center",
    fontsize=12
)

plt.tight_layout()
plt.show()


# ============================================================
# 7. COMPARISON TABLE
# ============================================================

comparison = {
    "Criteria": [
        "Image Sharpness",
        "Diversity",
        "Realism",
        "Training Stability",
        "Ease of Controlling Output",
        "Risk of Mode Collapse",
        "Typical Applications"
    ],

    "GAN": [
        "High",
        "High, but can decrease with mode collapse",
        "High",
        "Less Stable",
        "Moderate",
        "High",
        "Image generation and synthetic data"
    ],

    "VAE": [
        "Moderate / Slightly Blurry",
        "High",
        "Moderate",
        "More Stable",
        "Easy",
        "Very Low",
        "Anomaly detection and representation learning"
    ]
}

df = pd.DataFrame(comparison)

print("\nGAN vs VAE Comparison")
print("=" * 80)
print(df.to_string(index=False))


# ============================================================
# 8. SAVE COMPARISON TABLE
# ============================================================

df.to_csv("GAN_vs_VAE_Comparison.csv", index=False)

print("\nComparison table saved as:")
print("GAN_vs_VAE_Comparison.csv")


# ============================================================
# 9. FINAL VERDICT
# ============================================================

print("\nFINAL VERDICT")
print("=" * 80)

print("""
1. Synthetic Training Data:
   GAN is preferred because it generally produces sharper
   and more realistic images.

2. Anomaly Detection:
   VAE is preferred because it learns a useful latent
   representation and can identify abnormal samples using
   reconstruction error.

3. Overall:
   GAN is better for realistic image generation, while VAE
   is better for stable training, representation learning,
   and anomaly detection.
""")
