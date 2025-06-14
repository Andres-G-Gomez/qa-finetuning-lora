import json
import os
import glob
from pathlib import Path

def prompt_gpt2(prompt, model, tokenizer, max_tokens=50):
    # Move inputs to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Add this for inference
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def load_qa_data(data_dir):
    """Load QA pairs from all JSON files in the specified directory."""
    qa_texts = []
    data_path = Path(data_dir)
    
    # Find all JSON files in the directory
    json_files = list(data_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} JSON files to load")
    
    for json_file in json_files:
        print(f"Loading {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            # Convert QA pairs to training text format
            for qa in qa_pairs:
                # Format as Q: ... A: ... for clear structure
                formatted_text = f"Q: {qa['question']}\nA: {qa['answer']}"
                qa_texts.append(formatted_text)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Loaded {len(qa_texts)} QA pairs total")
    return qa_texts

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load QA dataset from files
    qa_data_dir = "./data/qa_pairs/"
    try:
        qa_texts = load_qa_data(qa_data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the QA data files are in the correct location.")
        exit(1)
    
    # Create dataset from QA pairs
    data = {"text": qa_texts}
    dataset = Dataset.from_dict(data)
    
    # Split dataset (80% train, 20% test)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    # 2. Load tokenizer
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token
    tokenizer.padding_side = "left"  # Important for generation

    # 3. Tokenize dataset with longer max_length for QA pairs
    def tokenize(example):
        # Use longer max_length to accommodate QA pairs
        tokens = tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=256  # Increased from 64 to handle longer QA content
        )
        # Add labels for causal language modeling
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True)
    
    # Remove the original text column to avoid conflicts
    tokenized = tokenized.remove_columns(["text"])

    # 4. Quantization config for QLoRA (only use if CUDA available)
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        # 5. Load quantized model and prepare for QLoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # For CPU, load model normally
        print("CUDA not available, loading model on CPU without quantization")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

    # 6. LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # Include both attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Training setup - adjusted for QA fine-tuning
    training_args = TrainingArguments(
        output_dir="./gpt2-qa-lora",
        per_device_train_batch_size=2,  # Reduced batch size due to longer sequences
        per_device_eval_batch_size=2,
        num_train_epochs=100,  # More epochs for better QA learning
        learning_rate=1e-4,  # Slightly higher learning rate
        logging_steps=10,
        save_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=device == "cuda",
        bf16=False,
        remove_unused_columns=True,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        warmup_steps=50,  # Add warmup for stability
        weight_decay=0.01,  # Add regularization
    )

    # 8. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8 if device == "cuda" else None
    )

    # 9. Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 10. Train
    print("Starting QA fine-tuning...")
    trainer.train()

    # 11. Save model
    print("Saving fine-tuned model...")
    model.save_pretrained("./gpt2-qa-lora")
    tokenizer.save_pretrained("./gpt2-qa-lora")

    # 12. Test the trained model with QA format
    print("\nTesting the fine-tuned QA model:")
    model.eval()
    
    # Test with QA format prompts
    test_questions = [
        "Q: What is the main purpose of health sector support projects?",
        "Q: Which regions typically benefit from health interventions?",
        "Q: What are the key components of health system strengthening?"
    ]
    
    for question in test_questions:
        print(f"\n{question}")
        print("Generated answer:")
        prompt_gpt2(question + "\nA:", model, tokenizer, max_tokens=100)
        print("-" * 50)
    
    print("QA fine-tuning completed successfully!")