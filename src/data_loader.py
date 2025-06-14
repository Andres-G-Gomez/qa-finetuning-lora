"""
Data loading utilities for QA fine-tuning project.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset


def load_qa_data(data_dir: str) -> List[str]:
    """
    Load QA pairs from all JSON files in the specified directory.
    
    Args:
        data_dir: Path to directory containing JSON files with QA pairs
        
    Returns:
        List of formatted QA text strings
        
    Raises:
        FileNotFoundError: If no JSON files found in directory
    """
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


def create_dataset(qa_texts: List[str], test_size: float = 0.2, seed: int = 42) -> Dict[str, Dataset]:
    """
    Create train/test dataset split from QA texts.
    
    Args:
        qa_texts: List of formatted QA text strings
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary containing 'train' and 'test' datasets
    """
    data = {"text": qa_texts}
    dataset = Dataset.from_dict(data)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    return dataset


def tokenize_dataset(dataset: Dict[str, Dataset], tokenizer, max_length: int = 256) -> Dict[str, Dataset]:
    """
    Tokenize the dataset for training.
    
    Args:
        dataset: Dictionary containing train/test datasets
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    def tokenize(example):
        tokens = tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        # Add labels for causal language modeling
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True)
    
    # Remove the original text column to avoid conflicts
    tokenized = tokenized.remove_columns(["text"])
    
    return tokenized