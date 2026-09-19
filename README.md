# ElevenceSkills Project

An internship project exploring text-to-image generation, from fine-tuning a pretrained
diffusion model down to building small conditional GANs and text-to-image pipelines from
scratch. The tasks progress from using existing foundation models (Task 1, 3) to designing,
training, and evaluating custom generative models (Task 2, 5, 6), with a dataset-analysis
task (Task 4) in between. A full write-up of methodology and results is in `report/`.

## Project Structure

```
ElevenceSkills-Project/
├── report/
│   └── ElevanceSkills_Internship_Report.docx   # Full written report
├── task1/    # LoRA fine-tuning of Stable Diffusion 1.5
├── task2/    # Conditional GAN (cGAN) for shape generation
├── task3/    # CLIP text encoding & caption similarity
├── task4/    # Dataset analysis (Oxford-102 Flowers)
├── task5/    # Self-Attention GAN (SAGAN) for shape generation
└── task6/    # End-to-end text-to-image generation pipeline
```

## Task Breakdown

### Task 1 — LoRA Fine-Tuning of Stable Diffusion
`task1/Task1_SD_LoRA_FineTunin.ipynb`

Fine-tunes **Stable Diffusion v1.5** on the `reach-vb/pokemon-blip-captions` dataset using
**LoRA** (Low-Rank Adaptation) via 🤗 `peft`, `diffusers`, and `transformers`. The UNet's
attention projections (`to_k`, `to_q`, `to_v`, `to_out.0`) are adapted with rank-4 LoRA
layers while the base VAE, text encoder, and UNet weights stay frozen, keeping the trainable
parameter count small.

- **Files:** notebook, `adapter_config.json`, `adapter_model.safetensors`,
  `pytorch_lora_weights.safetensors` (trained LoRA weights)
- **Run:** open the notebook in Colab/Jupyter with a GPU runtime; dependencies install in
  the first cell (`diffusers`, `transformers`, `accelerate`, `peft`, `datasets`)

### Task 2 — Conditional GAN (cGAN) for Shape Generation
`task2/cgan_new_G_conv.py`

A convolutional **class-conditional GAN** trained from scratch on a procedurally generated
dataset of four shapes (square, circle, triangle, rectangle) rendered with PIL. Includes the
full generator/discriminator, training loop, and sample outputs.

- **Files:** training script, trained generator weights (`cgan_new_G_conv.pth`),
  generated samples per class in `generated_samples/`
- **Run:** `python cgan_new_G_conv.py`

### Task 3 — CLIP Text Encoding & Caption Similarity
`task3/task3.py`

Uses a standalone **CLIP** text encoder (`openai/clip-vit-base-patch32`) to tokenize and
embed captions, then computes a cosine-similarity matrix between caption embeddings (e.g.
detecting that "a red square" and "a red square shape" are near-duplicates while unrelated
captions are not).

- **Files:** script, cached `text_embeddings.pt`, `similarity_heatmap.png`
- **Run:** `python task3.py`

### Task 4 — Dataset Analysis
`task4/task4_dataset_analysis.py`

Analyzes the **Oxford-102 Flowers** dataset: class counts and balance, image resolution
statistics, and auto-generated class descriptions, with visualizations of class distribution
and sample images.

- **Files:** script, `class_distribution.png`, `sample_images_with_descriptions.png`
- **Run:** `python task4_dataset_analysis.py`

### Task 5 — Self-Attention GAN (SAGAN) for Shape Generation
`task5/sagan_shapes.py`

A **class-conditional GAN with a self-attention block** in the generator, trained on the
same procedural shapes dataset (square, circle, triangle) at 32×32 resolution. The
self-attention lets image regions attend to other image regions (not to individual words).

- **Files:** script, trained generator weights (`sagan_generator.pth`), samples in `samples/`
- **Run:** `python sagan_shapes.py`

### Task 6 — Text-to-Image Generation Pipeline
`task6/pipeline script.py`

Combines the earlier pieces into an end-to-end **text-conditioned generation pipeline**:
CLIP encodes an input caption into a pooled embedding (using CLIP's `pooler_output`, not a
manual token average, to avoid diluting short captions with padding tokens), which
conditions a GAN generator with spectral normalization and a TTUR training schedule,
trained for 150 epochs. It supports free-text prompts paraphrasing the trained shape/class
vocabulary (e.g. "draw me a circle please").

- **Files:** script, trained generator weights (`task6_text_to_image_generator.pth`),
  cached caption embeddings (`task6_caption_cache.pt`), outputs in `pipeline_outputs/`
- **Run:** `pip install torch torchvision transformers diffusers pillow numpy scipy`
  then `python "pipeline script.py"`
- **Note:** the self-attention here is image-only (as in Task 5) — the model does not have
  word-level cross-attention like real Stable Diffusion, and the caption set consists of
  paraphrases of a small fixed vocabulary rather than open-vocabulary captions.

## Requirements

Core dependencies across tasks (install what each task needs, or all at once):

```
torch torchvision
diffusers transformers accelerate peft datasets
pillow numpy scipy matplotlib
```

A CUDA-capable GPU is recommended for Tasks 1, 2, 5, and 6 (training); Tasks 3 and 4 run
fine on CPU.

## Report

`report/ElevanceSkills_Internship_Report.docx` contains the full methodology, experiments,
and results write-up for all six tasks.
