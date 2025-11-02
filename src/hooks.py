"""Hook management for activation extraction."""
import torch
from collections import defaultdict
from typing import List, Callable
import logging

class ActivationHookManager:
    """Manage forward hooks for activation extraction."""
    
    def __init__(self):
        """Initialize hook manager."""
        self.hooks = []
        self.activations_storage = defaultdict(list)
    
    def create_hook(self, name: str) -> Callable:
        """
        Create a hook function for a specific layer.
        
        Args:
            name: Layer name identifier
            
        Returns:
            Hook function
        """
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations_storage[name].append(hidden_states.detach().cpu().half())
        return hook
    
    def register_hooks(self, modules: List[torch.nn.Module]):
        """
        Register hooks on all modules.
        
        Args:
            modules: List of modules to hook
        """
        print(f"Registering hooks on {len(modules)} modules...")
        
        for idx, module in enumerate(modules):
            hook_name = f"layer_{idx}"
            hook = module.register_forward_hook(self.create_hook(hook_name))
            self.hooks.append(hook)
        
        print(f"Registered {len(self.hooks)} hooks")
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations_storage.clear()
    
    def get_activation(self, layer_idx: int):
        """
        Get activation for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Activation tensor
        """
        layer_name = f"layer_{layer_idx}"
        if layer_name in self.activations_storage and len(self.activations_storage[layer_name]) > 0:
            return self.activations_storage[layer_name][0]
        return None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("All hooks removed")