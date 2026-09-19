import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = 64
NUM_CLASSES = 4  # 0: Square, 1: Circle, 2: Triangle, 3: Rectangle
LATENT_DIM = 100
EMBED_DIM = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCHS = 100
SAVE_INTERVAL = 10
SAVE_DIR = "outputs_new2"

# --- 1. Procedural Shape Dataset (Anti-aliased PIL rasterization) ---
from PIL import Image, ImageDraw

def draw_shape_image(shape_type, img_size=64):
    image = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(image)
    
    center_x, center_y = img_size // 2, img_size // 2
    size = random.randint(18, 30)
    
    # Slight positional jitter
    offset_x = random.randint(-4, 4)
    offset_y = random.randint(-4, 4)
    cx, cy = center_x + offset_x, center_y + offset_y
    
    if shape_type == 0:  # Square
        half = size // 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=255)
    elif shape_type == 1:  # Circle
        half = size // 2
        draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=255)
    elif shape_type == 2:  # Triangle
        p1 = (cx, cy - size // 2)
        p2 = (cx - size // 2, cy + size // 2)
        p3 = (cx + size // 2, cy + size // 2)
        draw.polygon([p1, p2, p3], fill=255)
    elif shape_type == 3:  # Rectangle
        half_w = int(size / 1.4)
        half_h = int(size / 2.2)
        draw.rectangle([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=255)
        
    img_array = np.array(image, dtype=np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)

class ShapeDataset2(Dataset):
    def __init__(self, num_samples_per_class=1000, img_size=64):
        self.samples = []
        for class_idx in range(NUM_CLASSES):
            for _ in range(num_samples_per_class):
                img = draw_shape_image(class_idx, img_size=img_size)
                self.samples.append((img, class_idx))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- 2. Model Weight Initialization ---
def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


# --- 3. Deep Convolutional CGAN Generator ---
class ConvGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        
        in_dim = LATENT_DIM + EMBED_DIM
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True)
        )
        
        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        img = self.deconv(x)
        return img

# --- 4. Deep Convolutional CGAN Discriminator (Feature Concatenation) ---
class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        
        self.img_conv = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        in_fc_dim = 128 * 8 * 8 + EMBED_DIM
        self.fc = nn.Sequential(
            nn.Linear(in_fc_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = self.img_conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, c], dim=1)
        validity = self.fc(x)
        return validity

# --- 5. Training Loop ---
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dataset = ShapeDataset2(num_samples_per_class=1000, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    netG = ConvGenerator().to(DEVICE)
    netD = ConvDiscriminator().to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    fixed_z = torch.randn(8, LATENT_DIM, device=DEVICE)
    fixed_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEVICE, dtype=torch.long)
    
    print(f"Starting Training for {EPOCHS} epochs on device: {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_len = imgs.size(0)
            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE, dtype=torch.long)
            
            real_targets = torch.full((batch_len, 1), 0.9, device=DEVICE)  # Real label smoothing
            fake_targets = torch.zeros((batch_len, 1), device=DEVICE)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            d_real_out = netD(real_imgs, labels)
            d_loss_real = criterion(d_real_out, real_targets)
            
            z = torch.randn(batch_len, LATENT_DIM, device=DEVICE)
            gen_imgs = netG(z, labels)
            d_fake_out = netD(gen_imgs.detach(), labels)
            d_loss_fake = criterion(d_fake_out, fake_targets)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_targets = torch.ones((batch_len, 1), device=DEVICE)
            g_out = netD(gen_imgs, labels)
            g_loss = criterion(g_out, g_targets)
            
            g_loss.backward()
            optimizer_G.step()
            
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            print(f"Epoch [{epoch}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            with torch.no_grad():
                sample_imgs = netG(fixed_z, fixed_labels)
                sample_imgs = (sample_imgs + 1.0) / 2.0  # Rescale to [0, 1]
                vutils.save_image(sample_imgs, f"{SAVE_DIR}/epoch_{epoch}.png", nrow=4)

    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'latent_dim': LATENT_DIM
    }, "cgan_new2_G.pth")
    print("Training finished! Saved model weights to 'cgan_new2_G.pth'.")

if __name__ == "__main__":
    train()
