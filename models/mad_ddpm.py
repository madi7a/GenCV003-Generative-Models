import torch
from torch import nn
import torch.nn.functional as F
import math


T = 200
IMG_SIZE = 32

def beta_scheduler(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

betas = beta_scheduler(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_index(vals, t, x_shape):
    batch_size = t.shape[0]  
    out = vals.to(t.device).gather(-1, t) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

@torch.no_grad()
def sample_timestep(model, x, t):
    """Takes a noisy image x at timestep t, predicts the noise, and takes one step backward."""
    device = x.device
    betas_t = get_index(betas, t, x.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = get_index(sqrt_one_minus_alphas_cumprod, t, x.shape).to(device)
    sqrt_recip_alphas_t = get_index(sqrt_recip_alphas, t, x.shape).to(device)
    
    predicted_noise = model(x, t)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t[0].item() == 0:
        return model_mean
    else:
        posterior_variance_t = get_index(posterior_variance, t, x.shape).to(device)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, -1).permute(0, 2, 1) 
        x_ln = self.ln(x_reshaped)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        out = (x_reshaped + attention_value).permute(0, 2, 1).view(B, C, H, W)
        return out

class MADUNet(nn.Module):
    def __init__(self, out_channels=3, time_emb_dim=256):
        super(MADUNet, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.time_emb1 = nn.Linear(time_emb_dim, 64)
        self.time_emb2 = nn.Linear(time_emb_dim, 128)
        self.time_emb3 = nn.Linear(time_emb_dim, 256)
        self.time_emb4 = nn.Linear(time_emb_dim, 512)
        self.time_emb5 = nn.Linear(time_emb_dim, 1024)

        # Downward Path
        self.l1_1 = nn.Conv2d(3,64,3,1,1)
        self.gn1_1 = nn.GroupNorm(8, 64)
        self.l1_2 = nn.Conv2d(64,64,3,1,1)
        self.gn1_2 = nn.GroupNorm(8, 64)

        self.l2_1 = nn.Conv2d(64,128,3,1,1)
        self.gn2_1 = nn.GroupNorm(8, 128)
        self.l2_2 = nn.Conv2d(128,128,3,1,1)
        self.gn2_2 = nn.GroupNorm(8, 128)

        self.l3_1 = nn.Conv2d(128,256,3,1,1)
        self.gn3_1 = nn.GroupNorm(8, 256)
        self.l3_2 = nn.Conv2d(256,256,3,1,1)
        self.gn3_2 = nn.GroupNorm(8, 256)

        self.l4_1 = nn.Conv2d(256,512,3,1,1)
        self.gn4_1 = nn.GroupNorm(8, 512)
        self.l4_2 = nn.Conv2d(512,512,3,1,1)
        self.gn4_2 = nn.GroupNorm(8, 512)

        # Bottleneck
        self.l5_1 = nn.Conv2d(512,1024,3,1,1)
        self.gn5_1 = nn.GroupNorm(8, 1024)
        self.attn = SelfAttention(1024) 
        self.l5_2 = nn.Conv2d(1024,1024,3,1,1)
        self.gn5_2 = nn.GroupNorm(8, 1024)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        # Upward Path
        self.up_conv1 = nn.ConvTranspose2d(1024,512,2,2)
        self.l_4  = nn.Conv2d(1024,512,3,1,1)
        self.gn4  = nn.GroupNorm(8, 512)
        self.l_42 = nn.Conv2d(512,512,3,1,1)
        self.gn42 = nn.GroupNorm(8, 512)
        
        self.up_conv2 = nn.ConvTranspose2d(512,256,2,2)
        self.l_3  = nn.Conv2d(512,256,3,1,1)
        self.gn3  = nn.GroupNorm(8, 256)
        self.l_32 = nn.Conv2d(256,256,3,1,1)
        self.gn32 = nn.GroupNorm(8, 256)
        
        self.up_conv3 = nn.ConvTranspose2d(256,128,2,2)
        self.l_2  = nn.Conv2d(256,128,3,1,1)
        self.gn2  = nn.GroupNorm(8, 128)
        self.l_22 = nn.Conv2d(128,128,3,1,1)
        self.gn22 = nn.GroupNorm(8, 128)
        
        self.up_conv4 = nn.ConvTranspose2d(128,64,2,2)
        self.l_1  = nn.Conv2d(128,64,3,1,1)
        self.gn1  = nn.GroupNorm(8, 64)
        self.l_12 = nn.Conv2d(64,64,3,1,1)
        self.gn12 = nn.GroupNorm(8, 64)

        self.fin = nn.Conv2d(64, out_channels, 1, 1)

    def forward(self, x, t): 
        t_emb = self.time_mlp(t)
        
        # Block 1
        x = self.l1_1(x)
        x = self.gn1_1(x)
        x = self.relu(x)
        x = x + self.time_emb1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.l1_2(x)
        x = self.gn1_2(x)
        x = self.relu(x)
        x1 = x
        x = self.pool(x)

        # Block 2
        x = self.l2_1(x)
        x = self.gn2_1(x)
        x = self.relu(x)
        x = x + self.time_emb2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.l2_2(x)
        x = self.gn2_2(x)
        x = self.relu(x)
        x2 = x
        x = self.pool(x)

        # Block 3
        x = self.l3_1(x)
        x = self.gn3_1(x)
        x = self.relu(x)
        x = x + self.time_emb3(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.l3_2(x)
        x = self.gn3_2(x)
        x = self.relu(x)
        x3 = x
        x = self.pool(x)

        # Block 4
        x = self.l4_1(x)
        x = self.gn4_1(x)
        x = self.relu(x)
        x = x + self.time_emb4(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.l4_2(x)
        x = self.gn4_2(x)
        x = self.relu(x)
        x4 = x
        x = self.pool(x)

        # Bottleneck
        x = self.l5_1(x)
        x = self.gn5_1(x)
        x = self.relu(x)
        x = x + self.time_emb5(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.attn(x) 
        x = self.l5_2(x)
        x = self.gn5_2(x)
        x = self.relu(x)

        # Up 1
        x = self.up_conv1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.l_4(x)
        x = self.gn4(x)
        x = self.relu(x)
        x = self.l_42(x)
        x = self.gn42(x)
        x = self.relu(x)

        # Up 2
        x = self.up_conv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.l_3(x)
        x = self.gn3(x)
        x = self.relu(x)
        x = self.l_32(x)
        x = self.gn32(x)
        x = self.relu(x)

        # Up 3
        x = self.up_conv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.l_2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.l_22(x)
        x = self.gn22(x)
        x = self.relu(x)

        # Up 4
        x = self.up_conv4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.l_1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.l_12(x)
        x = self.gn12(x)
        x = self.relu(x)

        return self.fin(x)