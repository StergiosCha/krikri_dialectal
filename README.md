# Greek Dialect LoRA Adapters - Clean Repository

This folder contains only the essential files needed for using the trained LoRA adapters. Training artifacts and large unnecessary files have been excluded.

## 📁 Structure

```
lora-dialects-clean/
├── .gitignore              # Git ignore rules
├── .gitattributes          # Git LFS configuration
├── GITHUB_SETUP.md         # GitHub setup instructions
├── lora-krikri-8b-base/    # LoRA adapter for Krikri-8B-Base
├── lora-llama3-8b-instruct/    # LoRA adapter for Llama-3-8B-Instruct
│   └── checkpoint-4000/    # Best checkpoint (eval_loss: 1.874)
└── lora-llama3.1-8b-instruct/ # LoRA adapter for Llama-3.1-8B-Instruct
```

## ✅ Included Files

Each adapter folder contains:
- `adapter_model.safetensors` - The trained LoRA weights (160MB)
- `adapter_config.json` - LoRA configuration
- `tokenizer.json` - Tokenizer files
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens mapping
- `chat_template.jinja` - Chat template for the model
- `README.md` - Model documentation

## ❌ Excluded Files

The following files were excluded to reduce repository size:

- **Training artifacts** (not needed for inference):
  - `optimizer.pt` (320MB each) - Optimizer state
  - `scheduler.pt` - Learning rate scheduler state
  - `rng_state.pth` - Random number generator state
  - `training_args.bin` - Training arguments

- **Large datasets**:
  - `*.jsonl` files (68MB+) - Training datasets

- **Logs and temporary files**:
  - `*.log` files
  - `*.out` files
  - Cache directories

- **Other checkpoints**:
  - Only the best checkpoint (checkpoint-4000) is included
  - Other checkpoints (3800, 4173) excluded to save space

## 📊 Size Comparison

- **Original folder**: ~2.3GB
- **Clean folder**: ~708MB
- **Reduction**: ~70% smaller

## 🚀 Usage

### Load a LoRA Adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",  # or your base model
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "./lora-llama3-8b-instruct"  # or path to other adapter
)

# Generate text
prompt = "Γράψε στην ποντιακή διάλεκτο: Καλημέρα!"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📝 Models

1. **lora-krikri-8b-base**
   - Base: `ilsp/Llama-Krikri-8B-Base`
   - Status: Training attempted (may have issues)

2. **lora-llama3-8b-instruct**
   - Base: `meta-llama/Meta-Llama-3-8B-Instruct`
   - Status: ✅ Fully trained (4,173 steps, 3 epochs)
   - Best checkpoint: checkpoint-4000 (eval_loss: 1.874)

3. **lora-llama3.1-8b-instruct**
   - Base: `meta-llama/Llama-3.1-8B-Instruct`
   - Status: ✅ Trained

## 🔧 GitHub Setup

See `GITHUB_SETUP.md` for instructions on:
- Setting up Git LFS for large files
- Pushing to GitHub
- Alternative: Using Hugging Face Hub

## 📦 Requirements

```bash
pip install transformers peft torch
```

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

[Add acknowledgments if needed]

