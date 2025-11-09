"""Dataset loading and processing."""
import os
import urllib
import zipfile
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig
import hydra
from dataset import download_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from metrics import compute_classification_metrics
from tqdm.auto import tqdm

def get_train_val_test_ids(size : int):
    arr = np.arange(size)
    train_ids, test_val_ids = train_test_split(arr, random_state = 42, test_size = 0.5)
    val_ids, test_ids = train_test_split(test_val_ids, random_state = 42, test_size = 0.5)
    return train_ids, val_ids, test_ids

def get_all_classifier_metrics(activations, targets, train_ids, val_ids, test_ids):
    train_activations, val_activations, test_activations = activations[train_ids], activations[val_ids], activations[test_ids]
    train_targets, val_targets, test_targets = targets[train_ids], targets[val_ids], targets[test_ids]

    logreg = LogisticRegression(random_state=42, max_iter = 1000)
    logreg.fit(train_activations, train_targets)
    
    test_preds = logreg.predict_proba(test_activations)[:, 1]
    val_preds = logreg.predict_proba(val_activations)[:, 1]
    metrics = compute_classification_metrics(
        test_targets, test_preds,
        y_val_true = val_targets, y_val_pred_proba = val_preds
    )
    return metrics


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    path_to_activations = config['path_to_activations']
    text_dataset = download_dataset(config)
    train_ids, val_ids, test_ids = None, None, None 
    layer2metrics=[]
    for layer_dir in tqdm(os.listdir(path_to_activations)): 
        path = Path(path_to_activations) / layer_dir
        num_batches = len(os.listdir(path))
        activations = np.concatenate([np.load(Path(path) / f"batch_{batch_id:04d}.npy") for batch_id in range(num_batches)])
        size = len(activations)
        if train_ids is None:
            train_ids, val_ids, test_ids = get_train_val_test_ids(size)
        targets = np.array([text_dataset[i][config.target_col] for i in range(len(text_dataset))]) 
        metrics = get_all_classifier_metrics(activations, targets, train_ids, val_ids, test_ids)
        layer2metrics.append(metrics)
    
    Path(os.path.dirname(config['path_to_save'])).mkdir(parents=True, exist_ok=True)
    with open(config['path_to_save'], 'w') as f:
        yaml.dump(layer2metrics, f)

if __name__ == '__main__':
    main()
