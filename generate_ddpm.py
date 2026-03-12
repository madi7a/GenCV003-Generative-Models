import torch
import torchvision.utils as vutils
from mad_ddpm import MADUNet, sample_timestep, T, IMG_SIZE


WEIGHTS_PATH = "models/mad_ddpm_cifar10.pth"
NUM_IMAGES = 16
OUTPUT_FILENAME = "ddpm_generated_grid.png"

def generate_ddpm_images():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model and Load Weights
    print("Loading MADUNet architecture and weights...")
    model = MADUNet(out_channels=3).to(device)
    
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find '{WEIGHTS_PATH}'. Please ensure it is in the same folder.")
        return

    model.eval()

    
    print(f"Starting reverse diffusion for {NUM_IMAGES} images over {T} timesteps...")
    with torch.no_grad():
        
        img = torch.randn((NUM_IMAGES, 3, IMG_SIZE, IMG_SIZE), device=device)
        
        
        for i in reversed(range(0, T)):
            t = torch.full((NUM_IMAGES,), i, device=device, dtype=torch.long)
            img = sample_timestep(model, img, t)
            
            
            if i % 50 == 0 or i == T - 1:
                print(f"Denoising... Step {i}/{T}")

        
        generated_images = (img + 1) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)

    
    vutils.save_image(generated_images, OUTPUT_FILENAME, nrow=4)
    print(f"Success! Images successfully saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate_ddpm_images()