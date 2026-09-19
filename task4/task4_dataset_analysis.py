from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ---------- 1. Load the dataset ----------
print("Loading Oxford-102 Flowers...")
dataset = load_dataset("nelorth/oxford-flowers", split="train")
print(dataset)

class_names = dataset.features["label"].names  # e.g. '1', '10', ... (numeric-string class IDs in this release)
num_classes = len(class_names)

# Build a simple natural-language description per class (dataset has no free-text captions)
def make_description(label_id):
    return f"a photo of flower species {class_names[label_id]}"

# ---------- 2. Dataset statistics ----------
labels = dataset["label"]
class_counts = Counter(labels)

print(f"\nNumber of classes: {num_classes}")
print(f"Total images: {len(dataset)}")
print(f"Images per class -> min: {min(class_counts.values())}, "
      f"max: {max(class_counts.values())}, "
      f"mean: {np.mean(list(class_counts.values())):.1f}")

# Image resolution stats (sample first N to keep it fast; full pass is fine too, just slower)
SAMPLE_N = 300
resolutions = []
for i in range(min(SAMPLE_N, len(dataset))):
    img = dataset[i]["image"]
    resolutions.append(img.size)  # (width, height)

widths, heights = zip(*resolutions)
print(f"\nImage resolution (sampled {len(resolutions)} images):")
print(f"  Width  -> min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"  Height -> min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")

# Description length stats (in words)
desc_lengths = [len(make_description(lbl).split()) for lbl in labels[:SAMPLE_N]]
print(f"\nDescription length (words, sampled {len(desc_lengths)}):")
print(f"  min: {min(desc_lengths)}, max: {max(desc_lengths)}, mean: {np.mean(desc_lengths):.1f}")

# ---------- 3. Visualize class distribution ----------
plt.figure(figsize=(10, 4))
plt.bar(range(num_classes), [class_counts.get(i, 0) for i in range(num_classes)])
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.title("Images per class - Oxford-102 Flowers")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
print("\nSaved class_distribution.png")

# ---------- 4. Display sample images with their descriptions ----------
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
sample_indices = np.random.choice(len(dataset), 8, replace=False)
for ax, idx in zip(axes.flatten(), sample_indices):
    item = dataset[int(idx)]
    ax.imshow(item["image"])
    ax.set_title(make_description(item["label"]), fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images_with_descriptions.png", dpi=150)
print("Saved sample_images_with_descriptions.png")