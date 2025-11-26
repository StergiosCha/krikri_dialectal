---
base_model: ilsp/Llama-Krikri-8B-Base
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:ilsp/Llama-Krikri-8B-Base
- lora
- transformers
---

# Model Card: Krikri Dialectal LoRA Adapter

A LoRA (Low-Rank Adaptation) fine-tuned adapter for generating text in Greek dialects. This model extends the [Llama-Krikri-8B-Base](https://huggingface.co/ilsp/Llama-Krikri-8B-Base) model with specialized training on natural prompts for four major Greek dialects: Pontic, Cretan, Northern Greek, and Cypriot.

## Model Details

### Model Description

This is a PEFT (Parameter-Efficient Fine-Tuning) adapter trained using LoRA on the Krikri-8B-Base model. The adapter enables the base model to generate text in Greek dialects when prompted with natural Greek instructions. The model was trained exclusively on dialectal data (excluding Standard Modern Greek) to focus on dialectal generation capabilities.

- **Developed by:** [Your Name/Organization]
- **Model type:** LoRA Adapter for Causal Language Modeling
- **Language(s) (NLP):** Greek (dialectal variants: Pontic, Cretan, Northern Greek, Cypriot)
- **License:** [Inherits from base model - specify if known]
- **Finetuned from model:** [ilsp/Llama-Krikri-8B-Base](https://huggingface.co/ilsp/Llama-Krikri-8B-Base)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

This model is designed for generating text in Greek dialects. It can be used for:

- **Dialectal text generation**: Generate text in Pontic, Cretan, Northern Greek, or Cypriot dialects
- **Dialectal conversation**: Engage in conversations using natural dialectal Greek
- **Cultural preservation**: Assist in documenting and preserving Greek dialectal varieties
- **Educational purposes**: Help learners understand dialectal variations in Greek

**Example prompts:**
- "Γράψε στην ποντιακή διάλεκτο: [your text]"
- "Απάντησε στα κρητικά: [your question]"
- "Στην κυπριακή γλώσσα: [your instruction]"

### Downstream Use

This adapter can be integrated into:
- Chatbots and conversational AI systems for dialectal Greek
- Text generation applications requiring dialectal output
- Language learning tools for Greek dialects
- Cultural heritage and documentation projects

### Out-of-Scope Use

This model is **not** suitable for:
- Standard Modern Greek generation (trained only on dialects)
- Formal or academic writing in dialects
- Real-time translation between dialects
- Medical, legal, or critical decision-making applications
- Generating content without human oversight

## Bias, Risks, and Limitations

### Limitations

- **Dialectal coverage:** Trained only on four major Greek dialects; may not perform well on other regional varieties
- **Standard Greek:** Explicitly excluded from training; not suitable for Standard Modern Greek generation
- **Data quality:** Performance depends on the quality and representativeness of training data
- **Dialectal variation:** Greek dialects have significant internal variation; model may not capture all sub-dialectal nuances
- **Context sensitivity:** May generate inappropriate or incorrect dialectal forms without proper context
- **Hallucination:** Like all language models, may generate plausible-sounding but incorrect dialectal text

### Risks

- **Cultural sensitivity:** Dialectal text generation should respect cultural and linguistic heritage
- **Misrepresentation:** Incorrect dialectal forms could mislead users about authentic dialectal usage
- **Bias:** Training data may reflect biases present in source materials

### Recommendations

Users should:
- Verify dialectal accuracy with native speakers or dialectal experts
- Use the model as a tool to assist, not replace, human expertise
- Be aware that generated text may require editing and fact-checking
- Consider the cultural and linguistic context when using dialectal generation
- Not use for critical applications without human oversight

## How to Get Started with the Model

### Installation

```bash
pip install transformers peft torch
```

### Basic Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "ilsp/Llama-Krikri-8B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ilsp/Llama-Krikri-8B-Base")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "your-username/your-model-name")

# Generate text
prompt = "Γράψε στην ποντιακή διάλεκτο: Καλημέρα, πώς είσαι;"
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

## Training Details

### Training Data

The model was trained on **23,421 natural prompt examples** covering four Greek dialects:
- **Pontic** (ποντιακή)
- **Cretan** (κρητική)
- **Northern Greek** (βορειοελλαδική)
- **Cypriot** (κυπριακή)

**Data preprocessing:**
- Artificial dialect tags (e.g., `<po>`, `<cr>`, `<no>`, `<cy>`) were converted to natural Greek prompts
- Standard Modern Greek examples were excluded from training
- Prompts use natural instructions like "Γράψε στην [dialect] διάλεκτο:" instead of tags
- Data was shuffled and split into train (95%) and validation (5%) sets

### Training Procedure

#### Preprocessing

1. Original dataset contained examples with artificial dialect tags
2. Tags were replaced with natural Greek prompt templates
3. Standard Greek examples were filtered out
4. Text was tokenized with max length of 512 tokens
5. Prompts and completions were concatenated for causal language modeling

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.1
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Number of epochs:** 3
- **Batch size:** 2 (per device)
- **Gradient accumulation steps:** 8 (effective batch size: 16)
- **Learning rate:** 3e-4
- **Warmup steps:** 100
- **Max sequence length:** 512 tokens
- **Trainable parameters:** 41,943,040 (0.51% of total model parameters)

#### Speeds, Sizes, Times

- **Adapter size:** ~160 MB (adapter_model.safetensors)
- **Base model size:** ~16 GB (8B parameters in bfloat16)
- **Total trainable parameters:** 41.9M (0.51% of base model)
- **Training timestamp:** 2025-09-09

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Validation split:** 5% of training data (approximately 1,171 examples)
- **Test strategy:** Held-out validation set during training

#### Factors

Evaluation should consider:
- **Dialect accuracy:** Correctness of dialectal forms
- **Dialect diversity:** Coverage across the four trained dialects
- **Prompt following:** Adherence to natural Greek instructions
- **Fluency:** Naturalness of generated dialectal text

#### Metrics

- **Loss:** Training and validation loss during fine-tuning
- **Perplexity:** Model's uncertainty in generating dialectal text
- **Human evaluation:** Recommended for assessing dialectal accuracy and naturalness

### Results

[Evaluation results to be added after assessment]

#### Summary

This adapter extends the Krikri-8B-Base model with specialized capabilities for generating text in Greek dialects. Performance should be evaluated through both automated metrics and human evaluation by native dialect speakers.



## Model Examination

The model uses LoRA adapters, which allow for efficient fine-tuning while maintaining the base model's general capabilities. The adapter weights can be examined separately from the base model, making it easier to understand what dialectal patterns the model has learned.

Key areas for examination:
- How the adapter modifies attention patterns for dialectal generation
- Which layers are most important for dialectal variation
- Comparison of adapter weights across different dialect prompts

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** GPU (specific model not available)
- **Training approach:** Parameter-efficient (LoRA) - only 0.51% of parameters trained
- **Training efficiency:** LoRA significantly reduces computational requirements compared to full fine-tuning
- **Carbon Emitted:** [To be calculated based on specific hardware and training duration]

## Technical Specifications

### Model Architecture and Objective

- **Base architecture:** Llama (8B parameters)
- **Adapter type:** LoRA (Low-Rank Adaptation)
- **Task:** Causal Language Modeling
- **Objective:** Generate dialectal Greek text given natural Greek prompts

### LoRA Configuration

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "bias": "none"
}
```

### Compute Infrastructure

#### Hardware

- Training performed on GPU infrastructure (specific hardware details not available)

#### Software

- **PEFT:** 0.17.1
- **Transformers:** (version from training environment)
- **PyTorch:** (version from training environment)
- **Python:** 3.12

## Citation

If you use this model, please cite:

**BibTeX:**

```bibtex
@misc{krikri-dialectal-lora,
  title={Krikri Dialectal LoRA Adapter: Fine-tuned Greek Dialect Generation},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://huggingface.co/your-username/your-model-name}}
}
```

**APA:**

[Your Name]. (2025). Krikri Dialectal LoRA Adapter: Fine-tuned Greek Dialect Generation [Computer software]. Hugging Face. https://huggingface.co/your-username/your-model-name

## Glossary

- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning technique that adds trainable low-rank matrices to model layers
- **PEFT (Parameter-Efficient Fine-Tuning):** A library for efficient fine-tuning methods that train only a small subset of model parameters
- **Dialectal Greek:** Regional varieties of Greek that differ from Standard Modern Greek in vocabulary, grammar, and pronunciation
- **Causal Language Modeling:** A task where the model predicts the next token given previous tokens

## More Information

This adapter was trained to preserve and generate Greek dialectal varieties. For questions, issues, or contributions, please open an issue on the model repository.

## Model Card Authors

[Your Name/Organization]

## Model Card Contact

[Your Contact Information]
### Framework versions

- PEFT 0.17.1