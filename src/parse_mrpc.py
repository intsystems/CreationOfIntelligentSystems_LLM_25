import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DownloadMode
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

token = '...'

# Configuration
#MODEL_NAME = "openlm-research/open_llama_3b"
#MODEL_NAME = "Qwen/Qwen2.5-3B"
#MODEL_NAME = 'MultiTrickFox/bloom-2b5_Zen'
#MODEL_NAME = "bigscience/bloom-7b1"
#MODEL_NAME = "bigscience/bloom-560m"

MODEL_NAME = "bigscience/bloom-1b1"

#OUTPUT_DIR = "activations_mrpc_openlllama3b"
#OUTPUT_DIR = "activations_mrpc_qwen2.5-3B"
#OUTPUT_DIR = 'activations_mrpc_bloom-2B5'
#OUTPUT_DIR = 'activations_mrpc_bloom-7b1'
#OUTPUT_DIR = "activations_mrpc_bloom-560m"
OUTPUT_DIR = 'activations_mrpc/activations_mrpc_bloom-1b1'
MAX_LENGTH = 512
BATCH_SIZE = 96
use_bloom = True

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Dataset class for MRPC
class MRPCDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"Sentence 1: {sample['sentence1']}\nSentence 2: {sample['sentence2']}"
        
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
            'idx': idx
        }

# Load model and tokenizer
print(f"Loading {MODEL_NAME} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    #cache_dir='/home/jovyan/rusakov/dim_lm/cache',
    token = token,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir='/home/jovyan/rusakov/dim_lm/cache',
    token = token,
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load MRPC dataset
print("Loading MRPC dataset...")
dataset = load_dataset(
    "glue", 
    "mrpc",
    # download_mode=DownloadMode.FORCE_REDOWNLOAD,
    cache_dir='/home/jovyan/rusakov/dim_lm/data'
)
train_data = dataset["train"]

# Get model info
if use_bloom:
    num_layers = len(model.transformer.h)
else:
    num_layers = len(model.model.layers)
hidden_dim = model.config.hidden_size
num_samples = len(train_data)

print(f"\nModel: {MODEL_NAME}")
print(f"Dataset: {num_samples} samples")
print(f"Layers: {num_layers}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")

# Create custom dataset and dataloader
mrpc_dataset = MRPCDataset(train_data, tokenizer, MAX_LENGTH)
dataloader = DataLoader(
    mrpc_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# Storage for activations per batch
activations_storage = defaultdict(list)

# Define hook function
def save_activation_hook(name):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activations_storage[name].append(hidden_states.detach().cpu().half())
    return hook

# Register hooks on all transformer blocks
print("Registering hooks on transformer blocks...")
hooks = []
if use_bloom:
    modules = [model.transformer.word_embeddings_layernorm] + list(model.transformer.h)
else:
    modules = [model.model.embed_tokens] + list(model.model.layers)

num_layers = len(modules)
# Create directory structure for each layer
print(f"\nCreating directory structure...")
for layer_idx in range(len(modules)):
    layer_dir = f"{OUTPUT_DIR}/layer_{layer_idx:02d}"
    os.makedirs(layer_dir, exist_ok=True)
print(f"Created {num_layers} layer directories in {OUTPUT_DIR}/")


for idx, layer in enumerate(modules):
    hook_name = f"layer_{idx}"
    hook = layer.register_forward_hook(save_activation_hook(hook_name))
    hooks.append(hook)

print(f"Registered hooks on {num_layers} transformer blocks")

# Run inference and collect activations
print(f"\nRunning batched inference (batch_size={BATCH_SIZE})...")
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

total_batches = batch_idx
print(f"\n✓ Processing complete!")
print(f"Batches saved: {total_batches}")
print(f"Files per layer: {total_batches}")
print(f"Total files: {total_batches * num_layers}")
print(f"Output directory: {OUTPUT_DIR}/")

# Verify with example file
example_file = f"{OUTPUT_DIR}/layer_00/batch_0000.npy"
example_act = np.load(example_file)
print(f"\nExample file: layer_00/batch_0000.npy")
print(f"Shape: {example_act.shape}")
print(f"Format: [batch_size={example_act.shape[0]}, seq_len={MAX_LENGTH}, hidden_dim={hidden_dim}]")
print(f"File size: {os.path.getsize(example_file) / (1024**2):.2f} MB")
print(f"Data type: {example_act.dtype}")

# Calculate total storage
total_size = 0
for layer_idx in range(num_layers):
    layer_dir = f"{OUTPUT_DIR}/layer_{layer_idx:02d}"
    layer_size = sum(os.path.getsize(f"{layer_dir}/{f}") for f in os.listdir(layer_dir) if f.endswith('.npy'))
    total_size += layer_size

print(f"\nTotal storage used: {total_size / (1024**3):.2f} GB")
print(f"Average per layer: {(total_size / num_layers) / (1024**3):.2f} GB")

# Print directory structure example
print(f"\nDirectory structure:")
print(f"{OUTPUT_DIR}/")
print(f"  ├── layer_00/")
print(f"  │   ├── batch_0000.npy  [{example_act.shape}]")
print(f"  │   ├── batch_0001.npy")
print(f"  │   └── ...")
print(f"  ├── layer_01/")
print(f"  │   └── ...")
print(f"  └── ...")
