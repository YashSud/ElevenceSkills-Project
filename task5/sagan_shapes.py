import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import random
import os

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

IMG_SIZE = 32
LABELS = ["square", "circle", "triangle"]
NUM_CLASSES = len(LABELS)
LATENT_DIM = 100
EMBED_DIM = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Dataset (same as task 2) ----------
def draw_shape(label_idx):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)
    margin = random.randint(4, 8)
    x0, y0 = margin, margin
    x1, y1 = IMG_SIZE - margin, IMG_SIZE - margin
    if label_idx == 0:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    elif label_idx == 1:
        draw.ellipse([x0, y0, x1, y1], fill=255)
    elif label_idx == 2:
        draw.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=255)
    return np.array(img, dtype=np.float32) / 127.5 - 1.0

def make_dataset(n_per_class=500):
    images, labels = [], []
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            images.append(draw_shape(cls))
            labels.append(cls)
    images = np.stack(images)[:, None, :, :]
    labels = np.array(labels)
    idx = np.random.permutation(len(images))
    return torch.tensor(images[idx]), torch.tensor(labels[idx])

# ---------- 2. Self-Attention block (SAGAN-style) ----------
class SelfAttention(nn.Module):
    """Lets every spatial position attend to every other position in the feature map."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts at 0: attention contributes nothing until learned useful

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)   # (B, N, C//8)
        k = self.key(x).view(B, -1, N)                       # (B, C//8, N)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)        # (B, N, N) - attention map

        v = self.value(x).view(B, C, N)                      # (B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

        return x + self.gamma * out, attn  # residual connection + return attention map for inspection

# ---------- 3. Generator & Discriminator with self-attention ----------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.project = nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4)
        self.up1 = nn.Sequential(
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 4x4 -> 8x8
        )
        self.up2 = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # 8x8 -> 16x16
        )
        self.attn = SelfAttention(64)  # placed at 16x16 - enough positions to matter, cheap enough to train fast
        self.up3 = nn.Sequential(
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),      # 16x16 -> 32x32
            nn.Tanh(),
        )

    def forward(self, z, labels, return_attention=False):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.project(x).view(-1, 256, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x, attn_map = self.attn(x)
        img = self.up3(x)
        if return_attention:
            return img, attn_map
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),                # 32x32 -> 16x16
            nn.LeakyReLU(0.2, True),
        )
        self.attn = SelfAttention(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),              # 16x16 -> 8x8
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),             # 8x8 -> 4x4
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, c], dim=1)
        x = self.conv1(x)
        x, _ = self.attn(x)
        return self.conv2(x)

# ---------- 4. Training ----------
def train(epochs=100, batch_size=64):
    images, labels = make_dataset()
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_imgs, real_labels in loader:
            bs = real_imgs.size(0)
            real_imgs, real_labels = real_imgs.to(DEVICE), real_labels.to(DEVICE)
            valid = torch.full((bs, 1), 0.9, device=DEVICE)
            fake = torch.zeros(bs, 1, device=DEVICE)

            opt_D.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            gen_imgs = G(z, gen_labels)
            d_real = criterion(D(real_imgs, real_labels), valid)
            d_fake = criterion(D(gen_imgs.detach(), gen_labels), fake)
            d_loss = (d_real + d_fake) / 2
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            g_loss = criterion(D(gen_imgs, gen_labels), torch.ones(bs, 1, device=DEVICE))
            g_loss.backward()
            opt_G.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f} | attn gamma: {G.attn.gamma.item():.4f}")

    return G

def generate(G, label_name, n=4):
    label_idx = LABELS.index(label_name)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        gen_labels = torch.full((n,), label_idx, dtype=torch.long, device=DEVICE)
        imgs = G(z, gen_labels).cpu().numpy()
    imgs = ((imgs + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return imgs

if __name__ == "__main__":
    print("Training self-attention CGAN on synthetic shapes...")
    G = train(epochs=100)
    torch.save(G.state_dict(), "sagan_generator.pth")
    print("Saved generator weights to sagan_generator.pth")

    os.makedirs("samples", exist_ok=True)
    for label_name in LABELS:
        imgs = generate(G, label_name, n=6)
        grid = np.concatenate([imgs[i, 0] for i in range(6)], axis=1)
        Image.fromarray(grid).save(f"samples/{label_name}.png")
        print(f"Saved samples/{label_name}.png")