"""
Task 6 - Comprehensive Text-to-Image Generation Pipeline (v3)

Changes from v2:
  1. FIXED BUG: encode_pooled() now uses CLIP's own `pooler_output` (its
     EOS-token pooling, trained for exactly this purpose) instead of a manual
     mean over all 77 token positions. The manual mean was diluting short
     captions like "a square" (4 real tokens) with ~73 padding-token
     embeddings averaged in alongside them. pooler_output doesn't have that
     problem - it's the standard way to get a single sentence-level vector
     out of a CLIP text encoder.
  2. Training set to 150 epochs (still TTUR + spectral norm from v2).
  3. Progress printed every 20 epochs instead of 10, to keep the log shorter
     over the longer run.

Report-accuracy notes (not bugs - just don't overclaim these in your writeup):
  - The self-attention block is IMAGE-only (same as Task 5): image regions
    attend to other image regions. There is no cross-attention where image
    regions attend to specific words - the text embedding is pooled into one
    vector before it reaches the generator, so the model can't "look at"
    individual words the way real Stable Diffusion cross-attention does.
    This is task-compliant (Task 5 explicitly allows self- OR cross-attention)
    - just describe it as image self-attention, not word-level attention.
  - The 6 caption templates per class are paraphrases of the same 3 shape
    words, not open-vocabulary captions. This demonstrates robustness to
    rephrasing ("draw me a circle please" still works), not generalization to
    concepts never trained on. Describe it that way in the report.

Run:
    pip install torch torchvision transformers diffusers pillow numpy scipy
    python task6_text_to_image_pipeline_v3.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Without these, cuDNN picks conv algorithms per-run for speed, which is not
# seeded - so the *same* script, same seed, can still produce a different
# (sometimes worse) generator run to run on GPU. This trades a little speed
# for reproducibility, which matters more here than raw throughput.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_SIZE = 32
LABELS = ["square", "circle", "triangle"]
NUM_CLASSES = len(LABELS)
LATENT_DIM = 100
EMBED_DIM = 50
CLIP_HIDDEN = 768
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_TEMPLATES = {
    "square":   ["a square", "a photo of a square", "a small square shape",
                 "an image of a square", "a plain white square", "a square outline"],
    "circle":   ["a circle", "a photo of a circle", "a small circle shape",
                 "an image of a circle", "a plain white circle", "a round circle outline"],
    "triangle": ["a triangle", "a photo of a triangle", "a small triangle shape",
                 "an image of a triangle", "a plain white triangle", "a triangle outline"],
}


# ---------------------------------------------------------------------------
# 1. Text preprocessing + embedding creation (Task 3, inlined)
# ---------------------------------------------------------------------------
class CLIPTextEncoderWrapper:
    def __init__(self, model_id=MODEL_ID, device=DEVICE):
        from transformers import CLIPTokenizer, CLIPTextModel
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device).eval()

    @torch.no_grad()
    def encode(self, prompts):
        """Per-token embeddings: [batch, seq_len, 768]. Used when you need the
        full sequence (e.g. real cross-attention, which this project doesn't use)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).to(self.device)
        return self.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]

    @torch.no_grad()
    def encode_pooled(self, prompts):
        """Single sentence-level vector per prompt: [batch, 768].
        FIX: uses CLIP's own pooler_output (EOS-token pooling) instead of a
        manual mean over all 77 positions - the manual mean averaged in ~73
        padding-token embeddings for a short caption like "a square", diluting
        the signal. pooler_output is built to avoid exactly that."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).to(self.device)
        output = self.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)
        return output.pooler_output

    def cache_pooled_embeddings(self, caption_templates, save_path=None):
        cache = {}
        for label, captions in caption_templates.items():
            cache[label] = self.encode_pooled(captions).cpu()
        if save_path:
            torch.save(cache, save_path)
        return cache


# ---------------------------------------------------------------------------
# 2. Shape dataset (unchanged)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Self-attention block (image-only - see report note at top of file)
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)
        k = self.key(x).view(B, -1, N)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out, attn


def weights_init(m):
    """DCGAN-style weight init. Left at PyTorch's defaults, this small a GAN
    converges inconsistently run to run - sometimes all 3 shapes come out
    clean, sometimes one collapses to a blob/blur, purely depending on the
    random draw of initial weights."""
    classname = m.__class__.__name__
    # spectral_norm (used throughout the Discriminator) renames the learnable
    # parameter to 'weight_orig' and recomputes 'weight' from it before every
    # forward call - so on those layers we must init weight_orig, or our init
    # gets silently discarded on the first forward pass.
    weight_param = "weight_orig" if hasattr(m, "weight_orig") else "weight"
    if classname.find("Conv") != -1:
        nn.init.normal_(getattr(m, weight_param).data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(getattr(m, weight_param).data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(getattr(m, weight_param).data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# 4. Generator (unchanged architecture) & Discriminator (spectral norm)
# ---------------------------------------------------------------------------
class TextConditionedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(CLIP_HIDDEN, EMBED_DIM), nn.ReLU(True))
        self.project = nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4)
        self.up1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.ConvTranspose2d(256, 128, 4, 2, 1))
        self.up2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1))
        self.attn = SelfAttention(64)
        self.up3 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z, text_embed, return_attention=False):
        c = self.text_proj(text_embed)
        x = torch.cat([z, c], dim=1)
        x = self.project(x).view(-1, 256, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x, attn_map = self.attn(x)
        img = self.up3(x)
        if return_attention:
            return img, attn_map
        return img


class TextConditionedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(CLIP_HIDDEN, IMG_SIZE * IMG_SIZE), nn.ReLU(True))
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True))
        self.attn = SelfAttention(64)
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 4 * 4, 1)), nn.Sigmoid(),
        )

    def forward(self, img, text_embed):
        c = self.text_proj(text_embed).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, c], dim=1)
        x = self.conv1(x)
        x, _ = self.attn(x)
        return self.conv2(x)


# ---------------------------------------------------------------------------
# 5. Training - TTUR + spectral norm, 150 epochs
# ---------------------------------------------------------------------------
def train(epochs=500, batch_size=64, embed_cache=None, log_every=20):
    images, labels = make_dataset()
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = TextConditionedGenerator().to(DEVICE)
    D = TextConditionedDiscriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))  # TTUR

    def sample_text_embeds(label_indices):
        out = []
        for idx in label_indices.tolist():
            variants = embed_cache[LABELS[idx]]
            pick = variants[random.randrange(variants.size(0))]
            out.append(pick)
        return torch.stack(out).to(DEVICE)

    for epoch in range(epochs):
        for real_imgs, real_labels in loader:
            bs = real_imgs.size(0)
            real_imgs, real_labels = real_imgs.to(DEVICE), real_labels.to(DEVICE)
            real_text = sample_text_embeds(real_labels)
            valid = torch.full((bs, 1), 0.9, device=DEVICE)
            fake = torch.zeros(bs, 1, device=DEVICE)

            opt_D.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            gen_text = sample_text_embeds(gen_labels)
            gen_imgs = G(z, gen_text)
            d_real = criterion(D(real_imgs, real_text), valid)
            d_fake = criterion(D(gen_imgs.detach(), gen_text), fake)
            d_loss = (d_real + d_fake) / 2
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            g_loss = criterion(D(gen_imgs, gen_text), torch.ones(bs, 1, device=DEVICE))
            g_loss.backward()
            opt_G.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | D loss: {d_loss.item():.4f} | "
                  f"G loss: {g_loss.item():.4f} | attn gamma: {G.attn.gamma.item():.4f}")

    return G


# ---------------------------------------------------------------------------
# 6. Inference + post-processing cleanup
# ---------------------------------------------------------------------------
def clean_up(img_uint8, median_size=3, threshold=128):
    smoothed = median_filter(img_uint8, size=median_size)
    binarized = np.where(smoothed > threshold, 255, 0).astype(np.uint8)
    return binarized


def generate_from_text(G, encoder, prompt, n=4, out_dir="pipeline_outputs", apply_cleanup=True):
    text_embed = encoder.encode_pooled([prompt] * n).to(DEVICE)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        imgs = G(z, text_embed).cpu().numpy()
    imgs = ((imgs + 1) * 127.5).clip(0, 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    panels = [imgs[i, 0] for i in range(n)]
    if apply_cleanup:
        panels = [clean_up(p) for p in panels]
    grid = np.concatenate(panels, axis=1)

    safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:40]
    path = f"{out_dir}/{safe_name}.png"
    Image.fromarray(grid).save(path)
    print(f"Prompt: {prompt!r:35s} -> saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Step 1: Loading CLIP text encoder (Task 3 component) ---")
    encoder = CLIPTextEncoderWrapper()

    print("\n--- Step 2: Caching pooled text embeddings (fixed pooler_output) ---")
    embed_cache = encoder.cache_pooled_embeddings(CAPTION_TEMPLATES, save_path="task6_caption_cache.pt")
    for label, emb in embed_cache.items():
        print(f"  {label}: {emb.shape[0]} caption variants cached, embedding dim {emb.shape[1]}")

    print("\n--- Step 3: Training text-conditioned self-attention GAN (150 epochs) ---")
    G = train(epochs=150, embed_cache=embed_cache, log_every=20)
    torch.save(G.state_dict(), "task6_text_to_image_generator.pth")
    print("Saved generator weights to task6_text_to_image_generator.pth")

    print("\n--- Step 4: Generating from free-form prompts (with cleanup post-processing) ---")
    demo_prompts = [
        "a square",
        "draw me a circle please",
        "a sharp triangle shape",
    ]
    for p in demo_prompts:
        generate_from_text(G, encoder, p, apply_cleanup=True)

    print("\nDone. Pipeline: text preprocessing -> text embedding (pooler_output) ->")
    print("GAN generation -> post-processing cleanup, all wired end-to-end.")