# GitHub Setup Guide

## ⚠️ Important: Large Files

This repository contains large model files (>100MB) that **cannot be pushed to GitHub directly**. You must use **Git LFS (Large File Storage)**.

## Setup Steps

### 1. Install Git LFS

```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs  # Debian/Ubuntu
# or
sudo yum install git-lfs      # RHEL/CentOS

# Windows
# Download from: https://git-lfs.github.com/
```

### 2. Initialize Git LFS in your repository

```bash
cd /Users/graogro/Dropbox/LORA_DIALECTAL
git lfs install
```

### 3. (Optional) Remove unnecessary large files before committing

The following files are training artifacts and not needed for inference:
- `optimizer.pt` files (320MB each) in checkpoint folders
- `scheduler.pt` files
- `rng_state.pth` files

You can remove them to save space:

```bash
# Remove optimizer states from checkpoints
find lora-* -name "optimizer.pt" -delete
find lora-* -name "scheduler.pt" -delete
find lora-* -name "rng_state.pth" -delete
```

### 4. Initialize Git repository (if not already done)

```bash
git init
git add .gitignore .gitattributes
git commit -m "Add .gitignore and Git LFS configuration"
```

### 5. Add files with Git LFS

```bash
# Git LFS will automatically handle files matching patterns in .gitattributes
git add .
git commit -m "Add LoRA adapters for Greek dialects"
```

### 6. Push to GitHub

```bash
# Create repository on GitHub first, then:
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

## File Size Summary

- **Adapter models**: 160MB each (3 models = 480MB)
- **Checkpoint adapters**: 160MB each (3 checkpoints = 480MB)
- **Optimizer states**: 320MB each (3 files = 960MB) - **RECOMMENDED TO DELETE**
- **Total without optimizers**: ~1GB
- **Total with optimizers**: ~2GB

## What Gets Tracked

✅ **Included (via Git LFS):**
- `adapter_model.safetensors` files
- `*.bin` model files
- Tokenizer files

❌ **Excluded (via .gitignore):**
- `optimizer.pt` files (training artifacts)
- `scheduler.pt` files
- `rng_state.pth` files
- Large dataset files (`*.jsonl`)
- Log files
- Cache directories

## Alternative: Use Hugging Face Hub

Instead of GitHub, consider uploading models to Hugging Face Hub:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload a model
huggingface-cli upload your-username/your-model-name ./lora-llama3-8b-instruct
```

This is better for:
- Large model files
- Model versioning
- Easy model sharing
- No file size limits

## Notes

- GitHub free tier includes 1GB Git LFS storage and 1GB/month bandwidth
- For larger repositories, consider GitHub Pro or Hugging Face Hub
- Checkpoint folders can be kept for training resumption, but optimizer states are not needed for inference

