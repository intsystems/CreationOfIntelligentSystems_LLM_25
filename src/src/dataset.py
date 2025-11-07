"""Dataset loading and processing."""
import os
import urllib
import zipfile
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

class Enwik8LlamaDataset(Dataset):
    """Custom Dataset for enwik8 with proper tokenization"""
    
    def __init__(self, data, tokenizer, max_length=512, stride=256):
        """
        Args:
            data: raw text string
            tokenizer: HuggingFace tokenizer (e.g., LlamaTokenizer)
            max_length: maximum sequence length for the model
            stride: stride for sliding window (if None, no overlap)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride else max_length
        
        self.data = data
        
        self.encodings = tokenizer(
            self.data,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None
        )
        self.input_ids = self.encodings['input_ids']
        
        # Calculate number of samples based on sliding window
        self.num_samples = max(1, (len(self.input_ids) - max_length) // self.stride + 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Calculate start position with stride
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length
        
        # Handle the last sample
        if end_idx > len(self.input_ids):
            start_idx = len(self.input_ids) - self.max_length
            end_idx = len(self.input_ids)
        
        # Get input sequence
        input_ids = self.input_ids[start_idx:end_idx]
        
        # Create attention mask (all 1s since we have real tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'idx':idx
        }

def preprocess_enwik8(raw_path, clean=True):
    """Read and optionally clean enwik8 data"""
    with open(raw_path, 'rb') as f:
        data = f.read()
    
    # Convert to string
    data = data.decode('utf-8', errors='ignore')
    
    if clean:
        # Remove XML/HTML tags (basic cleaning)
        import re
        # Remove XML tags
        data = re.sub(r'<[^>]+>', '', data)
        # Remove extra whitespace
        data = re.sub(r'\s+', ' ', data)
    
    return data

def download_enwik8(config):
    """Download enwik8 dataset from Matt Mahoney's site"""

    data_dir = os.path.join(config.dataset.cache_dir)

    os.makedirs(data_dir, exist_ok=True)
    
    url = 'http://mattmahoney.net/dc/enwik8.zip'
    zip_path = os.path.join(data_dir, 'enwik8.zip')
    raw_path = os.path.join(data_dir, 'enwik8')
    
    # Download if not exists
    if not os.path.exists(raw_path):
        print(f'Downloading enwik8 from {url}...')
        urllib.request.urlretrieve(url, zip_path)
        
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        print('Download complete!')
    return preprocess_enwik8(raw_path)

def download_dataset(config: DictConfig):
    """
    Load dataset.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Dataset split
    """
    print(f"Loading {config.dataset.name}/{config.dataset.subset} dataset...")
    
    if config.dataset.name == 'other' and config.dataset.subset == 'enwik8':
        return download_enwik8(config)
    dataset = load_dataset(
        config.dataset.name,
        config.dataset.subset,
        cache_dir=config.dataset.cache_dir
    )
    
    split_data = dataset[config.dataset.split]
    print(f"Loaded {len(split_data)} samples from {config.dataset.split} split")
    
    return split_data