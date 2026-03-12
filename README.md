# GenCV003: Generative Models (DDPM & VAE)

## Project Overview
This repository contains pure PyTorch implementations of two generative models built entirely from scratch: a Denoising Diffusion Probabilistic Model (DDPM) and a Convolutional Variational Autoencoder (ConvVAE). Both models are trained to generate novel images, with primary benchmarking done on the CIFAR-10 dataset (32x32 resolution).

## Architectures Implemented
1. **MADUNet (DDPM):** A custom U-Net architecture featuring Group Normalization, Sinusoidal Positional Timestep Embeddings, and a Self-Attention bottleneck to capture global spatial context.
2. **ConvMADVAE (VAE):** A Convolutional Variational Autoencoder that natively compresses and expands 2D spatial features using `Conv2d` and `ConvTranspose2d` layers to prevent the loss of spatial locality. 

## Quantitative Results (Fréchet Inception Distance)
FID was used to measure the statistical similarity between the generated images and the real datasets (Lower is better).
* **DDPM on CIFAR-10 (20 Epochs):** 81.10
* **VAE on CIFAR-10 (30 Epochs):** 144.06

## Bonus: MNIST Latent Space Generalization
To further validate the structural integrity of the `ConvMADVAE` architecture, the model was also adapted and trained on the MNIST dataset (rescaled to 32x32). It successfully generalized to the high-contrast structural features, achieving an **FID of 56.19** in just 10 epochs. 

## How to Reproduce the Results

### 1. Environment Setup
Install the required dependencies:
`pip install -r requirements.txt`

### 2. One-Click Inference (Generate Images)
You do not need to train the models from scratch. Pre-trained weights are provided in the `weights/` folder. Two generation scripts are located in the root directory for instant inference.

**To generate images using the DDPM:**
`python generate_ddpm.py`

**To generate images using the VAE:**
`python generate_vae.py`

*Note: Inside `generate_vae.py` and `generate_ddpm.py`, there are commented lines allowing you to easily swap between generating the CIFAR-10 models or the MNIST model. Simply uncomment the targeted weight path.*

### 3. Training from Scratch (Notebooks)
If you wish to view the training loops, data transformations, and FID calculation scripts, all original execution code is organized within the `notebooks/` directory.

### 4. Detailed Evaluation Report
For a complete breakdown of the architectural trade-offs, qualitative differences (e.g., the VAE "blur" vs. the DDPM noise iterations), and the engineering pivots made during development, please refer to the `GenCV003_Report.pdf` included in this repository.