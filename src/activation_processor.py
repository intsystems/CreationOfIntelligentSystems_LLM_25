"""Main processing logic for activation extraction."""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
import logging

from dataset import MRPCDataset, load_glue_dataset, SSTDataset, QQPDataset
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
        dataset = load_glue_dataset(self.config)
        print(f"Dataset samples: {len(dataset)}")
        
        # Create dataset and dataloader
        dataset_class_mapping = {
            'mrpc':MRPCDataset,
            'sst2':SSTDataset,
            'qqp':QQPDataset
        }
        dataset = dataset_class_mapping[self.config.dataset.subset](
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
                aggregated = aggregate_activations(
                    layer_activations_np,
                    batch['attention_mask'].cpu().numpy(),
                    self.config.aggregation.method,
                    self.config.aggregation.use_attention_mask
                )
                
                # Save to file
                layer_dir = Path(self.output_dir) / f"layer_{layer_idx:02d}"
                filename = layer_dir / f"batch_{batch_idx:04d}.npy"
                np.save(filename, aggregated)
            
            batch_idx += 1
        
        # Cleanup
        self.hook_manager.remove_hooks()