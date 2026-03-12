

# ======================================================================
# ======================================================================
# Bonus: MNIST Generalization
# As a sanity check for the latent space, the model was trained on MNIST. 
# Due to the high contrast and lower structural complexity of digits, 
# the VAE achieved a superior FID of **56.19** in only 10 epochs.
# =====================================================================
# =====================================================================

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================
IMG_SIZE = 32
BATCH_SIZE = 64
Z_DIM = 256
EPOCHS = 10 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==========================================
# 2. THE MNIST DATALOADER (The Magic Trick)
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),          # Scale 28x28 up to 32x32
    transforms.Grayscale(num_output_channels=3),      # Convert 1 channel to 3 channels!
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)          # Scale to [-1, 1]
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=data_transforms)
data = torch.utils.data.ConcatDataset([train_data, test_data])
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 3. YOUR CONV-VAE ARCHITECTURE
# ==========================================
class ConvMADVAE(nn.Module):
    def __init__(self, z_dim=256):
        super(ConvMADVAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)

        # Decoder
        self.dec_fc = nn.Linear(z_dim, 1024)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) 
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1) 
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 256, 2, 2) 
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        return torch.tanh(self.dec_conv4(x)) 

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z_repara = mu + (std * epsilon)
        x_recon = self.decode(z_repara)
        return x_recon, mu, logvar

vae_model = ConvMADVAE(z_dim=Z_DIM).to(device)

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (beta * kld_loss)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
optimizer = Adam(vae_model.parameters(), lr=1e-3)

print("Starting MNIST VAE Training...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x = batch[0].to(device)
        x_recon, mu, logvar = vae_model(x)
        
        loss = vae_loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
            
    print(f"--- Epoch {epoch+1}/{EPOCHS} | Average Loss: {epoch_loss/len(dataloader):.2f} ---")

# ==========================================
# 5. GENERATE AND PLOT
# ==========================================
vae_model.eval()
with torch.no_grad():
    random_latent_vectors = torch.randn(10, Z_DIM).to(device)
    generated_images = vae_model.decode(random_latent_vectors)

generated_images = (generated_images.cpu() + 1) / 2.0
generated_images = torch.clamp(generated_images, 0.0, 1.0)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("MNIST Generated by ConvVAE", fontsize=16)
for i, ax in enumerate(axes.flatten()):
    img = generated_images[i].permute(1, 2, 0).numpy()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.show()

# ==========================================
# 6. CALCULATE FID SCORE
# ==========================================
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

@torch.no_grad()
def generate_fake_vae_batch(batch_size):
    random_latent_vectors = torch.randn(batch_size, Z_DIM).to(device)
    generated_images = vae_model.decode(random_latent_vectors)
    return generated_images

print("Calculating FID over 32 batches...")
test_iterator = iter(dataloader)
for i in range(32):
    # Process Real
    try:
        real_images = next(test_iterator)[0].to(device)
    except StopIteration:
        test_iterator = iter(dataloader)
        real_images = next(test_iterator)[0].to(device)
        
    real_images = torch.clamp((real_images + 1) / 2.0, 0.0, 1.0)
    fid.update(real_images, real=True)
    
    # Process Fake
    fake_images = generate_fake_vae_batch(BATCH_SIZE)
    fake_images = torch.clamp((fake_images + 1) / 2.0, 0.0, 1.0)
    fid.update(fake_images, real=False)

fid_score = fid.compute()
print(f"\nFinal MNIST VAE FID Score: {fid_score.item():.2f}")

# ==========================================
# 7. SAVE THE MNIST MODEL
# ==========================================
mnist_vae_save_path = 'vae_mnist_weights.pth'
torch.save(vae_model.state_dict(), mnist_vae_save_path)
print(f"MNIST VAE weights successfully saved to: {mnist_vae_save_path}")