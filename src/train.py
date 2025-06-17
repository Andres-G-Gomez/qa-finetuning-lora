"""
Main training script for QA fine-tuning with QLoRA.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from data_loader import load_qa_data, create_dataset, tokenize_dataset


def load_config(config_path: str = "../configs/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_model_and_tokenizer(config: dict, device: str):
    """Setup model, tokenizer, and LoRA configuration."""
    model_id = config['model']['model_id']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Setup quantization config for CUDA
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization']['load_in_4bit'],
            bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("CUDA not available, loading model on CPU without quantization")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def setup_training_args(config: dict, device: str) -> TrainingArguments:
    """Setup training arguments from config."""
    training_config = config['training']
    
    return TrainingArguments(
        output_dir=training_config['output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        num_train_epochs=training_config['num_train_epochs'],
        learning_rate=training_config['learning_rate'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_strategy=training_config['eval_strategy'],
        save_strategy=training_config['save_strategy'],
        logging_dir=training_config['logging_dir'],
        fp16=device == "cuda",
        bf16=False,
        remove_unused_columns=training_config['remove_unused_columns'],
        report_to=training_config['report_to'],
        dataloader_pin_memory=training_config['dataloader_pin_memory'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
    )


def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading QA data...")
    qa_texts = load_qa_data(config['data']['data_dir'])
    dataset = create_dataset(
        qa_texts, 
        test_size=config['data']['test_size'], 
        seed=config['data']['seed']
    )
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config, device)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(
        dataset, 
        tokenizer, 
        max_length=config['model']['max_length']
    )
    
    # Setup training arguments
    training_args = setup_training_args(config, device)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=config['data_collator']['mlm'],
        pad_to_multiple_of=config['data_collator']['pad_to_multiple_of'] if device == "cuda" else None
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting QA fine-tuning...")
    trainer.train()
    
    # Save model
    print("Saving LoRA adapters...")
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    print("LoRA adapters saved!")
    print("QA fine-tuning completed successfully!")


if __name__ == "__main__":
    main()