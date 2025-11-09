"""Model loading utilities."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertLMHeadModel, OPTForCausalLM
from omegaconf import DictConfig
import logging

def get_torch_dtype(dtype_str: str):
    """Convert string to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_model_and_tokenizer(config: DictConfig):
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model.name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
        token=config.model.token if config.model.token else None,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=get_torch_dtype(config.model.torch_dtype),
        device_map=config.model.device_map,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
        token=config.model.token if config.model.token else None,
    )
    model.eval()
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    print(f"Model loaded on device: {model.device}")
    
    return model, tokenizer


def get_model_layers(model, use_bloom: bool):
    """
    Get transformer layers from model.
    
    Args:
        model: HuggingFace model
        use_bloom: Whether model is BLOOM architecture
        
    Returns:
        List of modules to hook
    """
    if use_bloom:
        modules = [model.transformer.word_embeddings_layernorm] + list(model.transformer.h)
    elif isinstance(model, BertLMHeadModel):
        modules = [model.bert.embeddings.word_embeddings] + list(model.bert.encoder.layer)
    elif isinstance(model, OPTForCausalLM):
        modules = [model.model.decoder.embed_tokens] + list(model.model.decoder.layers)
    else:
        modules = [model.model.embed_tokens] + list(model.model.layers)
    
    return modules


def get_model_info(model, use_bloom: bool):
    """
    Get model architecture information.
    
    Args:
        model: HuggingFace model
        use_bloom: Whether model is BLOOM architecture
        
    Returns:
        Dictionary with model info
    """
    modules = get_model_layers(model, use_bloom)
    
    return {
        'num_layers': len(modules),
        'hidden_dim': model.config.hidden_size,
        'modules': modules
    }