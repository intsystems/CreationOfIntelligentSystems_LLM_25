import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DownloadMode
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import pickle
import hydra
from omegaconf import DictConfig

class TopicDataset(Dataset):
        def __init__(self, path_to_topic_pkl : str, is_positive, tokenizer, max_length : int):
            with open(path_to_topic_pkl, 'rb') as f:
                data = pickle.load(f)
                pos_neg_str = 'positive' if is_positive else 'negative'
                self.topic_texts = data[list(data.keys())[0]][pos_neg_str]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.topic_texts)

        def __getitem__(self, index):
            text = self.topic_texts[index]
            encoding = self.tokenizer(
                text,
                max_length = self.max_length,
                truncation = True,
                padding = 'max_length',
                return_tensors = 'pt'
            )
            
            return {
                'input_ids':encoding['input_ids'].squeeze(0),
                'attention_mask':encoding['attention_mask'].squeeze(0),
                'idx':index
            }

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig):
    TOPIC_NAME = config['topic']
    IS_POSITIVE: bool = config['is_positive']

    PATH_TO_TOPIC = config['path_to_topic']
    pos_neg_str = 'positive' if IS_POSITIVE else 'negative'
    OUTPUT_DIR = config['output_dir'] 

    MODEL_NAME = "openlm-research/open_llama_3b"
    MAX_LENGTH = config['max_length']
    BATCH_SIZE = config['batch_size']
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        cache_dir='/home/jovyan/rusakov/dim_lm/cache',
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir='/home/jovyan/rusakov/dim_lm/cache',
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset = TopicDataset(
        PATH_TO_TOPIC,
        is_positive = IS_POSITIVE,
        tokenizer = tokenizer,
        max_length = MAX_LENGTH
    )
    hidden_dim = model.config.hidden_size
    num_samples = len(train_dataset)
    num_layers = len(model.model.layers)

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )

    activations_storage = defaultdict(list)

    # Define hook function
    def save_activation_hook(name):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            activations_storage[name].append(hidden_states.detach().cpu().half())
        return hook

    modules = [model.model.embed_tokens] + list(model.model.layers)
    num_layers = len(modules)

    for layer_idx in range(len(modules)):
        layer_dir = f"{OUTPUT_DIR}/layer_{layer_idx:02d}"
        os.makedirs(layer_dir, exist_ok=True)

    hooks = []

    for idx, layer in enumerate(modules):
        hook_name = f"layer_{idx}"
        hook = layer.register_forward_hook(save_activation_hook(hook_name))
        hooks.append(hook)

    batch_idx = 0

    for batch in tqdm(dataloader, desc="Processing batches"):
        activations_storage.clear()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Save activations for each layer separately
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx}"
            # Shape: [batch_size, seq_len, hidden_dim]
            layer_activations = activations_storage[layer_name][0].numpy()
            
            # Save to corresponding layer directory
            layer_dir = f"{OUTPUT_DIR}/layer_{layer_idx:02d}"
            filename = f"{layer_dir}/batch_{batch_idx:04d}.npy"
            
            layer_activation_agg = np.sum(layer_activations*batch['attention_mask'].unsqueeze(-1).cpu().numpy(), axis=1)
            layer_activation_agg /= (batch['attention_mask'].unsqueeze(-1).cpu().numpy().sum(axis = 1) + 1e-9)

            np.save(filename, layer_activation_agg)
        
        batch_idx += 1

    # Remove hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    main()