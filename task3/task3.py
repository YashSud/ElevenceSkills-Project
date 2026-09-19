import torch
from transformers import CLIPTokenizer, CLIPTextModel

MODEL_ID = "openai/clip-vit-base-patch32"  # standalone public CLIP checkpoint (same family SD's text encoder uses)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Load tokenizer + text encoder ----------
print("Loading tokenizer and text encoder...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID).to(DEVICE)
text_encoder.eval()
print(f"Loaded. Vocab size: {tokenizer.vocab_size}, embedding dim: {text_encoder.config.hidden_size}")


def preprocess(captions, max_length=None):
    """Text descriptions -> tokenized representation (input_ids + attention_mask)."""
    if max_length is None:
        max_length = tokenizer.model_max_length
    tokens = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokens.input_ids.to(DEVICE), tokens.attention_mask.to(DEVICE)


def encode_text(captions):
    """Text descriptions -> encoded embeddings ready for a text-to-image model."""
    input_ids, attention_mask = preprocess(captions)
    with torch.no_grad():
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return {
        "input_ids": input_ids,                        # (batch, seq_len) token IDs
        "token_embeddings": output.last_hidden_state,   # (batch, seq_len, 768) - per-token, used by cross-attention in SD/GANs
        "pooled_embedding": output.pooler_output,       # (batch, 768) - one vector per caption
    }


def cosine_sim_matrix(pooled_embeddings):
    normed = torch.nn.functional.normalize(pooled_embeddings, dim=-1)
    return normed @ normed.T


if __name__ == "__main__":
    captions = [
        "a photo of a red square",
        "a picture of a red square shape",     # near-duplicate of caption 0 -> should be very similar
        "a photo of a blue circle",
        "a fluffy orange cat sleeping",
        "an astronaut riding a horse on the moon",
    ]

    print("\n--- Step 1: Tokenization ---")
    input_ids, attention_mask = preprocess(captions)
    print("input_ids shape:", input_ids.shape)          # (5, 77) - CLIP's fixed sequence length
    print("attention_mask shape:", attention_mask.shape)
    print("Tokens for caption 0:", tokenizer.convert_ids_to_tokens(input_ids[0])[:12], "...")

    print("\n--- Step 2: Encoding ---")
    result = encode_text(captions)
    print("token_embeddings shape:", result["token_embeddings"].shape)  # (5, 77, 768)
    print("pooled_embedding shape:", result["pooled_embedding"].shape)  # (5, 768)

    print("\n--- Step 3: Verify embeddings capture meaning ---")
    sim = cosine_sim_matrix(result["pooled_embedding"])
    print("Cosine similarity matrix:")
    for i, cap in enumerate(captions):
        row = " ".join(f"{v:.2f}" for v in sim[i].tolist())
        print(f"  [{row}]  <- \"{cap[:40]}\"")

    # Save embeddings + heatmap for the report
    torch.save(result, "text_embeddings.pt")
    print("\nSaved embeddings to text_embeddings.pt")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim.cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(captions)))
    ax.set_yticks(range(len(captions)))
    ax.set_xticklabels([f"cap {i}" for i in range(len(captions))], rotation=45)
    ax.set_yticklabels([f"cap {i}" for i in range(len(captions))])
    for i in range(len(captions)):
        for j in range(len(captions)):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, label="cosine similarity")
    plt.title("Caption embedding similarity")
    plt.tight_layout()
    plt.savefig("similarity_heatmap.png", dpi=150)
    print("Saved similarity_heatmap.png")