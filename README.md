# Greek Dialect LoRA Adapters

LoRA (Low-Rank Adaptation) fine-tuned adapters for generating text in Greek dialects. These adapters extend base language models with specialized training on Greek dialectal data.

## 📁 Models

1. **lora-krikri-8b-base**
   - Base Model: `ilsp/Llama-Krikri-8B-Base`
   - Trained for Greek dialect generation

2. **lora-llama3-8b-instruct**
   - Base Model: `meta-llama/Meta-Llama-3-8B-Instruct`
   - Fully trained (4,173 steps, 3 epochs)
   - Best checkpoint: `checkpoint-4000` (eval_loss: 1.874)

3. **lora-llama3.1-8b-instruct**
   - Base Model: `meta-llama/Llama-3.1-8B-Instruct`
   - Trained for Greek dialect generation

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

## 📦 Requirements

```bash
pip install transformers peft torch
```

## 📄 License

This work is licensed for **research purposes only**. You are free to use, modify, and distribute these models for research purposes.

## 🙏 Acknowledgments

**Computing Resources:**
- AWS resources were provided by the [National Infrastructures for Research and Technology (GRNET)](https://www.grnet.gr/)
- Funded by the [EU Recovery and Resiliency Facility](https://ec.europa.eu/info/business-economy-euro/recovery-coronavirus/recovery-and-resilience-facility_en)

**Base Models:**
- [Llama-Krikri-8B-Base](https://huggingface.co/ilsp/Llama-Krikri-8B-Base) by ILSP
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) by Meta
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) by Meta

## 📝 Model Details

Each adapter folder contains:
- `adapter_model.safetensors` - The trained LoRA weights
- `adapter_config.json` - LoRA configuration
- `tokenizer.json` - Tokenizer files
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens mapping
- `chat_template.jinja` - Chat template for the model

## 🔧 Technical Details

- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** All attention and MLP projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
