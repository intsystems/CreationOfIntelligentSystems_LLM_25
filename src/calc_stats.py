import numpy as np
import os
from scipy.linalg import svd
from functools import wraps
from collections import defaultdict
from tqdm.auto import tqdm
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

from intrinsics_dimension import twonn_pytorch

def parse_layer(layer_num : int, path_to_activations : str):
    path = f'{path_to_activations}/layer_{layer_num:02d}'
    batches = []
    for filename in os.listdir(path):
        filepath = f"{path}/{filename}"
        # [bs, emd_dim]
        batch = np.load(filepath)
        batches.append(batch)
    # [N, emb_dim]
    batches = np.concatenate(batches)
    return batches

def add_layer_parsing_and_averaging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        layer_num, path_to_activations, num_samples = kwargs['layer_num'], kwargs['path_to_activations'], kwargs['num_samples']
        assert 'layer_num' in kwargs and 'path_to_activations' in kwargs
        del kwargs['path_to_activations']
        del kwargs['layer_num']
        
        X = parse_layer(layer_num=layer_num, path_to_activations=path_to_activations)

        if 'num_estimates' not in kwargs or kwargs['num_estimates'] is None:
            if 'num_samples' in kwargs:
                del kwargs['num_samples']
            return func(X, *args, **kwargs)
        
        num_estimates = kwargs['num_estimates']
        del kwargs['num_estimates']
        del kwargs['num_samples']

        estim_value : float = 0
        for _ in range(num_estimates):
            ids = np.random.randint(0, len(X), num_samples)
            estim_value += func(X[ids], *args, **kwargs)
        estim_value /= num_estimates

        return estim_value

    return wrapper

@add_layer_parsing_and_averaging
def parse_anisotropy_svd(X,
                         center = False):
    if center:
        X_centered = X - np.mean(X, axis=0, keepdims=True)
    else:
        X_centered = X
    
    _, singular_values, _ = svd(X_centered, full_matrices=False)
    
    sigma_squared = singular_values ** 2
    anisotropy = sigma_squared[0] / np.sum(sigma_squared)
    
    return float(anisotropy)

@add_layer_parsing_and_averaging
def parse_twonn_dimention(X):
    return float(twonn_pytorch(torch.Tensor(X)))

@add_layer_parsing_and_averaging
def parse_singular_dimention(X,
                             variance_threshold=0.90):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    variance = S ** 2
    total_variance = np.sum(variance)
    cumulative_variance = np.cumsum(variance) / total_variance
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    return float(k)


# example usage: calculate, using all activations from layer 1
# parse_anisotropy_svd(layer_num = 1,
#                     path_to_activations = '......../qwen2.5-3b',
#                     num_samples = -1,
#                     center = False)
#
#
# example usage: calculate, random 928 acrivations and avergate over 16 plays
# parse_anisotropy_svd(layer_num = 1,
#                     path_to_activations = '......../qwen2.5-3b',
#                     num_samples = 928,
#                     num_estimates = 16,
#                     center = False)

def calculate_and_save_stats(
        path_to_activations : str,
        path_to_save : str,
        num_samples : int = None,
        num_estimates : int = None
):
    stats = defaultdict(list)
    for i in tqdm(range(len(os.listdir(path_to_activations)))):
        stats['twoNN_dim'].append(
            parse_twonn_dimention(layer_num = i,
                                  path_to_activations = path_to_activations,
                                  num_samples = num_samples,
                                  num_estimates = num_estimates)
            )
        stats['anisotropy'].append(
            parse_anisotropy_svd(layer_num = i,
                                  path_to_activations = path_to_activations,
                                  num_samples = num_samples,
                                  num_estimates = num_estimates,
                                  center=False)
            )
        stats['singular_dim_0.9'].append(
            parse_singular_dimention(layer_num = i,
                                  path_to_activations = path_to_activations,
                                  num_samples = num_samples,
                                  num_estimates = num_estimates,
                                  variance_threshold=0.9)
            )
        stats['singular_dim_0.99'].append(
            parse_singular_dimention(layer_num = i,
                                  path_to_activations = path_to_activations,
                                  num_samples = num_samples,
                                  num_estimates = num_estimates,
                                  variance_threshold=0.99)
            )
        stats['singular_dim_0.999'].append(
            parse_singular_dimention(layer_num = i,
                                  path_to_activations = path_to_activations,
                                  num_samples = num_samples,
                                  num_estimates = num_estimates,
                                  variance_threshold=0.999)
            )
    with open(path_to_save, 'w') as f:
        yaml.dump(dict(stats), f)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    num_samples = None if 'num_samples' not in config else config['num_samples']
    num_estimates = None if 'num_estimates' not in config else config['num_estimates']
    calculate_and_save_stats(
        path_to_activations = config['path_to_activations'],
        path_to_save = config['path_to_save'],
        num_samples = num_samples,
        num_estimates = num_estimates
    )
if __name__ == '__main__':
    main()