import json
import random
from collections import defaultdict

def convert_to_natural_prompts_dialects_only():
    """Convert artificial tags to natural Greek prompts - DIALECTS ONLY, NO STANDARD GREEK"""
    
    prompt_templates = {
        'pontic': [
            "Γράψε στην ποντιακή διάλεκτο:",
            "Απάντησε στα ποντιακά:",
            "Στην ποντιακή γλώσσα:",
            "Συνέχισε στα ποντιακά:",
        ],
        'cretan': [
            "Γράψε στην κρητική διάλεκτο:",
            "Απάντησε στα κρητικά:",
            "Στην κρητική γλώσσα:",
            "Συνέχισε στα κρητικά:",
        ],
        'northern': [
            "Γράψε στη βορειοελλαδική διάλεκτο:",
            "Απάντησε στα βορειοελλαδικά:",
            "Στη διάλεκτο της Βόρειας Ελλάδας:",
            "Συνέχισε στα βορειοελλαδικά:",
        ],
        'cypriot': [
            "Γράψε στην κυπριακή διάλεκτο:",
            "Απάντησε στα κυπριακά:",
            "Στην κυπριακή γλώσσα:",
            "Συνέχισε στα κυπριακά:",
        ]
    }
    
    examples_by_dialect = defaultdict(list)
    
    with open('dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'prompt' in data and 'completion' in data:
                    prompt = data['prompt']
                    completion = data['completion']
                    
                    # ONLY process dialects - SKIP <gr>
                    if prompt.startswith('<po>'):
                        dialect = 'pontic'
                        clean_prompt = prompt[4:].strip()
                    elif prompt.startswith('<cr>'):
                        dialect = 'cretan'
                        clean_prompt = prompt[4:].strip()
                    elif prompt.startswith('<no>'):
                        dialect = 'northern'
                        clean_prompt = prompt[4:].strip()
                    elif prompt.startswith('<cy>'):
                        dialect = 'cypriot'
                        clean_prompt = prompt[4:].strip()
                    else:
                        continue  # Skip <gr> and anything else
                    
                    template = random.choice(prompt_templates[dialect])
                    
                    if clean_prompt:
                        natural_prompt = f"{template} {clean_prompt}"
                    else:
                        natural_prompt = template
                    
                    examples_by_dialect[dialect].append({
                        'prompt': natural_prompt,
                        'completion': completion
                    })
                    
            except json.JSONDecodeError:
                continue
    
    # Show statistics
    for dialect, examples in examples_by_dialect.items():
        print(f"{dialect}: {len(examples)} examples")
    
    # Combine all dialect examples
    all_examples = []
    for dialect_examples in examples_by_dialect.values():
        all_examples.extend(dialect_examples)
    
    random.shuffle(all_examples)
    
    # Save dialects-only natural prompts dataset
    with open('natural_prompts_dialects_only.jsonl', 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\nConverted {len(all_examples)} DIALECT-ONLY examples to natural prompts")
    return len(all_examples)

if __name__ == "__main__":
    convert_to_natural_prompts_dialects_only()
