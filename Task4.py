import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from torchvision import transforms

# Pretty much the entirety of this code, shamefully, was AI generated.
# It works (barely).
class BrainPNGDataset(Dataset):
    def __init__(self, folder_path, image_size=(64, 64)):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        self.transform = transforms.Compose([
            transforms.Grayscale(),        # Convert to 1-channel
            transforms.Resize(image_size),
            transforms.ToTensor()          # [0,255] → [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        return self.transform(img)

# Setup the datasets.
train_dataset = BrainPNGDataset("keras_png_slices_data/keras_png_slices_train")
val_dataset   = BrainPNGDataset("keras_png_slices_data/keras_png_slices_validate")
test_dataset  = BrainPNGDataset("keras_png_slices_data/keras_png_slices_test")

# Load the data.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# AI wrote this for fun???
def load_2d_slices_from_nii(path, slice_axis=2, slice_range=(30, 100)):
    """Load 2D slices from a 3D MRI scan (png format)"""
    img = Image.open(path)
    data = img.get_fdata()
    slices = []

    for i in range(slice_range[0], slice_range[1]):
        if slice_axis == 0:
            slices.append(data[i, :, :])
        elif slice_axis == 1:
            slices.append(data[:, i, :])
        else:
            slices.append(data[:, :, i])
    return np.array(slices)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)# 16x16 -> 8x8
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)   # 32x32 -> 64x64

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.sigmoid(self.deconv3(x))

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(50):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # print("Train samples:", len(train_dataset))
    # print("Val samples:", len(val_dataset))
    # print("Test samples:", len(test_dataset))
    val_loss = validate_epoch(model, val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        recon, _, _ = model(batch)
        break  # Only one batch

# Visualize first 5

for i in range(5):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(batch[i].cpu().squeeze(), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(recon[i].cpu().squeeze(), cmap='gray')
    axs[1].set_title('Reconstructed')
    plt.show()
