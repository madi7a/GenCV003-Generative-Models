import torch
import torchvision.utils as vutils
from mad_vae import ConvMADVAE 

#  to generate cifer10 data
WEIGHTS_PATH = "/weights/vae_cifar10_weights.pth" 
# to generate mnsit data
# WEIGHTS_PATH = "/weights/vae_mnist_weights.pth"
NUM_IMAGES = 16
Z_DIM = 256
OUTPUT_FILENAME = "vae_generated_grid.png"

def generate_images():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Initialize Model and Load Weights
    print("Loading VAE architecture and weights...")
    model = ConvMADVAE(z_dim=Z_DIM).to(device)
    
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find '{WEIGHTS_PATH}'. Please ensure your .pt file is in the same folder.")
        return

   
    model.eval()

    #Generate from pure noise
    print(f"Generating {NUM_IMAGES} new images from latent space...")
    with torch.no_grad():
        random_noise = torch.randn(NUM_IMAGES, Z_DIM).to(device)
        generated_images = model.decode(random_noise)

        # Scale from [-1, 1] back to [0, 1] for saving
        generated_images = (generated_images + 1) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)


    vutils.save_image(generated_images, OUTPUT_FILENAME, nrow=4)
    print(f"Success! Images successfully saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate_images()