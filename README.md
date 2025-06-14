# QA Fine-tuning with QLoRA

A parameter-efficient fine-tuning project that adapts GPT-2 for domain-specific question-answering using QLoRA (Quantized Low-Rank Adaptation).

## Project Overview

This project demonstrates how to fine-tune large language models efficiently for specialized QA tasks. Using QLoRA, we achieve strong performance while using minimal computational resources and storage.

**Key Features:**

- Parameter-efficient fine-tuning with QLoRA
- 4-bit quantization for memory optimization
- Domain-specific QA adaptation (Health Sector Projects)
- Complete training and inference pipeline
- Modular, production-ready code structure

## Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended) or CPU
# 8GB+ GPU memory or 16GB+ RAM
```

### Installation

```bash
git clone https://github.com/Andres-G-Gomez/qa-finetuning-lora.git
cd qa-finetuning-lora

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Training

```bash
# Add your QA JSON files to data/qa_pairs/
# Then run training
chmod +x scripts/run_training.sh
./scripts/run_training.sh
```

## Results

| Metric            | Target    | Achieved |
| ----------------- | --------- | -------- |
| Final Eval Loss   | 1.2-1.8   | TBD      |
| Training Time     | ~30 min   | TBD      |
| Model Size (LoRA) | ~10-20 MB | TBD      |
| GPU Memory Usage  | ~8 GB     | TBD      |

### Sample Outputs

```
Q: What is the main purpose of health sector support projects?
A: Health sector support projects aim to strengthen healthcare systems by improving infrastructure, building capacity, and enhancing service delivery to achieve better health outcomes for populations.

Q: Which regions typically benefit from health interventions?
A: Health interventions typically target underserved regions with poor health indicators, including rural areas, conflict-affected zones, and regions with limited healthcare infrastructure.
```

## Architecture

- **Base Model:** GPT-2 (124M parameters)
- **Fine-tuning Method:** QLoRA with 4-bit quantization
- **LoRA Config:** r=8, alpha=16, dropout=0.05
- **Target Modules:** c_attn, c_proj
- **Training Data:** Health sector QA pairs

## Project Structure

```
qa-finetuning-lora/
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── data/
│   └── qa_pairs/            # QA training data (JSON files)
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluation scripts
│   └── inference.py        # Model loading and inference
├── configs/
│   └── training_config.yaml # Training parameters
├── results/
│   ├── training_logs/       # Training output logs
│   └── evaluation_metrics/  # Evaluation results
├── scripts/
│   ├── setup.sh            # Environment setup
│   └── run_training.sh     # Training pipeline
├── logs/                   # Training logs
└── gpt2-qa-lora/          # Saved LoRA adapters
```

## Usage

### Training Configuration

Edit `configs/training_config.yaml` to adjust:

- Model parameters (max_length, model_id)
- LoRA settings (rank, alpha, dropout)
- Training hyperparameters (epochs, learning rate, batch size)

### Running Training

```bash
# Method 1: Use training script
./scripts/run_training.sh

# Method 2: Direct execution
cd src
python train.py
```

### Inference

```bash
# Test with sample questions
python src/inference.py --test

# Ask a specific question
python src/inference.py --question "What are the benefits of health system strengthening?"

# Programmatic usage
from src.inference import QAInference
qa_model = QAInference("./gpt2-qa-lora")
answer = qa_model.generate_answer("Your question here")
```

### Evaluation

```bash
# Evaluate on test data
python src/evaluate.py --save_results

# Evaluate specific number of samples
python src/evaluate.py --sample_size 50 --save_results
```

## Technical Details

### Training Configuration

- **Epochs:** 5 (configurable)
- **Batch Size:** 2 per device
- **Learning Rate:** 2e-4
- **Max Sequence Length:** 256 tokens
- **Quantization:** 4-bit NF4 (CUDA only)

### Data Format

QA pairs should be in JSON format:

```json
[
  {
    "question": "What is the purpose of health sector support?",
    "answer": "Health sector support aims to strengthen healthcare systems..."
  }
]
```

### Hardware Requirements

- **Minimum:** 8GB GPU memory or 16GB RAM (CPU)
- **Recommended:** 16GB+ GPU memory (RTX 3080/4080)
- **Training Time:** ~20-40 minutes depending on data size

## Monitoring Training

Monitor training progress:

- **Target eval loss:** 1.2-1.8
- **Watch for:** Decreasing eval loss without overfitting
- **Stop if:** Eval loss starts increasing consistently

Logs are saved to `logs/training_TIMESTAMP.log`

## Experimentation

### Hyperparameter Tuning

Key parameters to experiment with in `configs/training_config.yaml`:

- **LoRA rank (r):** 4, 8, 16, 32
- **Learning rate:** 1e-4, 2e-4, 5e-4
- **Batch size:** 1, 2, 4
- **Epochs:** 3, 5, 10
- **Max length:** 128, 256, 512

### Performance Tips

1. **Increase LoRA rank** for better adaptation (more parameters)
2. **Adjust learning rate** if loss plateaus or diverges
3. **Use larger batch size** if you have more GPU memory
4. **Increase max_length** for longer QA pairs

## Key Learnings

- QLoRA enables efficient fine-tuning on consumer hardware
- Domain-specific QA requires careful prompt formatting ("Q: ... A: ...")
- Parameter-efficient methods maintain base model knowledge while adapting
- Proper evaluation metrics are crucial for assessing performance
- Modular code structure enables easier experimentation and deployment

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient Finetuning of Quantized LLMs
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index) - Parameter-Efficient Fine-Tuning
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].

---

_Built with PyTorch, Transformers, and PEFT • Optimized for efficiency and modularity_
