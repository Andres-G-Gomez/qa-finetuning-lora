# QA Fine-tuning with QLoRA

A parameter-efficient fine-tuning project that adapts GPT-2 for domain-specific question-answering using QLoRA (Quantized Low-Rank Adaptation).

## Project Overview

This project demonstrates how to fine-tune large language models efficiently for specialized QA tasks. Using QLoRA, we achieve strong performance while using minimal computational resources and storage.

**Key Features:**

- Parameter-efficient fine-tuning with QLoRA
- 4-bit quantization for memory optimization
- Domain-specific QA adaptation
- Complete training and inference pipeline

## Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended) or CPU
```

### Installation

```bash
git clone https://github.com/yourusername/qa-finetuning-lora.git
cd qa-finetuning-lora
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

## Results

| Metric            | Value      |
| ----------------- | ---------- |
| Final Eval Loss   | X.XX       |
| Training Time     | XX minutes |
| Model Size (LoRA) | ~XX MB     |
| GPU Memory Usage  | ~XX GB     |

### Sample Outputs

```
Q: What is the main purpose of health sector support projects?
A: [Your model's generated answer here]
```

## Architecture

- **Base Model:** GPT-2 (124M parameters)
- **Fine-tuning Method:** QLoRA with 4-bit quantization
- **LoRA Config:** r=8, alpha=16, dropout=0.05
- **Target Modules:** c_attn, c_proj

## Project Structure

```
├── data/
│   └── qa_pairs/          # QA training data
├── train.py               # Main training script
├── requirements.txt       # Dependencies
├── README.md             # This file
└── gpt2-qa-lora/         # Saved LoRA adapters
```

## Technical Details

### Training Configuration

- **Epochs:** 5
- **Batch Size:** 2
- **Learning Rate:** 2e-4
- **Max Sequence Length:** 256
- **Quantization:** 4-bit NF4

### Hardware Requirements

- **Minimum:** 8GB GPU memory or 16GB RAM (CPU)
- **Recommended:** 16GB+ GPU memory
- **Training Time:** ~XX minutes on RTX 3080

## Monitoring Training

The model saves checkpoints and logs training progress. Monitor eval loss to prevent overfitting:

- **Target eval loss:** 1.2-1.8
- **Stop if:** eval loss increases (overfitting)

## Usage

### Loading the Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./gpt2-qa-lora")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-qa-lora")

# Generate answer
prompt = "Q: Your question here\nA:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Experimentation

### Hyperparameter Tuning

Key parameters to experiment with:

- LoRA rank (r): 4, 8, 16, 32
- Learning rate: 1e-4, 2e-4, 5e-4
- Batch size: 1, 2, 4
- Epochs: 3, 5, 10

## Key Learnings

- QLoRA enables efficient fine-tuning on consumer hardware
- Domain-specific QA requires careful prompt formatting
- Parameter-efficient methods maintain base model knowledge while adapting to new tasks

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

---

_Built with PyTorch, Transformers, and PEFT_
