"""Main processing logic for activation extraction."""
import time
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
import logging

from dataset import MRPCDataset, download_dataset, SSTDataset, QQPDataset, Enwik8LlamaDataset, WritingPromptsDataset, MATH_500Dataset, MATH_500wPromptDataset
from model import load_model_and_tokenizer, get_model_info, get_model_layers
from hooks import ActivationHookManager
from utils import create_output_structure, aggregate_activations, get_storage_stats

class ActivationProcessor:
    """Process model activations and save to disk."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize processor.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataloader = None
        self.hook_manager = ActivationHookManager()
        self.output_dir = None
        self.num_layers = None
        
    def setup(self):
        """Setup model, data, and output structure."""
        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.config)
        
        # Get model info
        model_info = get_model_info(self.model, self.config.model.use_bloom)
        self.num_layers = model_info['num_layers']
        
        print(f"Model: {self.config.model.name}")
        print(f"Layers: {model_info['num_layers']}")
        print(f"Hidden dimension: {model_info['hidden_dim']}")
        
        # Load dataset
        dataset = download_dataset(self.config)
        print(f"Dataset samples: {len(dataset)}")
        
        # Create dataset and dataloader
        dataset_class_mapping = {
            'mrpc':MRPCDataset,
            'sst2':SSTDataset,
            'qqp':QQPDataset,
            'enwik8': Enwik8LlamaDataset,
            'euclaise/writingprompts': WritingPromptsDataset,
            'HuggingFaceH4/MATH-500': MATH_500wPromptDataset if self.config.dataset.prompt else MATH_500Dataset,
        }
        key = self.config.dataset['subset'] if 'subset' in self.config.dataset else self.config.dataset.name
        dataset = dataset_class_mapping[key](
            dataset,
            self.tokenizer,
            self.config.dataset.max_length
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.processing.batch_size,
            shuffle=self.config.processing.shuffle,
            num_workers=self.config.processing.num_workers
        )
        
        print(f"Batch size: {self.config.processing.batch_size}")
        print(f"Total batches: {len(self.dataloader)}")
      
        # Create output structure
        self.output_dir = create_output_structure(
            self.config.output.base_dir,
            self.config.output.model_subdir,
            self.num_layers
        )
        
        # Register hooks
        modules = get_model_layers(self.model, self.config.model.use_bloom)
        self.hook_manager.register_hooks(modules)
    
    def process(self):
        """Run inference and save activations."""
        batch_idx = 0
        
        for batch in tqdm(self.dataloader, desc="Processing batches"):
            self.hook_manager.clear_activations()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Save activations for each layer
            for layer_idx in range(self.num_layers):
                layer_activations = self.hook_manager.get_activation(layer_idx)
                
                if layer_activations is None:
                    print(f"No activations for layer {layer_idx}")
                    continue
                
                # Convert to numpy
                layer_activations_np = layer_activations.numpy()
                
                # Aggregate activations
                methods = [self.config.aggregation.method]
                if isinstance(self.config.aggregation.method, ListConfig):
                    methods = self.config.aggregation.method
                for method in methods:
                    aggregated = aggregate_activations(
                        layer_activations_np,
                        batch['attention_mask'].cpu().numpy(),
                        method,
                        self.config.aggregation.use_attention_mask
                    )
                
                    # Save to file
                    layer_dir = Path(self.output_dir) /f'agg_{method}'/ f"layer_{layer_idx:02d}"
                    Path(layer_dir).mkdir(parents=True, exist_ok=True)
                    filename = layer_dir / f"batch_{batch_idx:04d}.npy"
                    np.save(filename, aggregated)
                    if 'target' in batch:
                        targets_dir = Path(self.output_dir) / 'targets'
            
            batch_idx += 1
        
        # Cleanup
        self.hook_manager.remove_hooks()