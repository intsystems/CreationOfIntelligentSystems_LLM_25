"""Dataset loading and processing."""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig

class MRPCDataset(Dataset):
    """Custom Dataset class for MRPC."""
    
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_length: int):
        """
        Initialize MRPC dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"Are these 2 sentences equivalent: {sample['sentence1']} and {sample['sentence2']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx,
            'target':sample['label']
        }
    
class SSTDataset(Dataset):
    """Custom Dataset class for SST."""
    
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_length: int):
        """
        Initialize MRPC dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"Determine if the following sentence is positive or negative: {sample}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx,
            'target':sample['label']
        }

class QQPDataset(Dataset):
    """Custom Dataset class for MRPC."""
    
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_length: int):
        """
        Initialize MRPC dataset.
        
        Args:
            data: HuggingFace dataset
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"Are these 2 questions equivalent: {sample['question1']} and {sample['question2']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx,
            'target':sample['label']
        }



def load_glue_dataset(config: DictConfig):
    """
    Load GLUE dataset.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Dataset split
    """
    print(f"Loading {config.dataset.name}/{config.dataset.subset} dataset...")
    
    dataset = load_dataset(
        config.dataset.name,
        config.dataset.subset,
        cache_dir=config.dataset.cache_dir
    )
    
    split_data = dataset[config.dataset.split]
    print(f"Loaded {len(split_data)} samples from {config.dataset.split} split")
    
    return split_data