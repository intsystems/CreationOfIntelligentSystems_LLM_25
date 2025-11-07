import numpy as np
import torch
from intrinsics_dimension import twonn_pytorch
import os

SAVE_STATS_PATH = 'all_stats/openLlama3B_topic_Bird'
DIRNAME = 'activations_topics/open_llama_3b/Bird'

def parse_layer(layer_num : int):
    path = f'/home/jovyan/rusakov/dim_lm/{DIRNAME}/layer_{layer_num:02d}'
    batches = []
    for filename in os.listdir(path):
        filepath = f"{path}/{filename}"
        # [bs, emd_dim]
        batch = np.load(filepath)
        #batch = np.load(filepath)[:, -1, :] # last hidden from seq_len
        batches.append(batch)
    # [N, emb_dim]
    batches = np.concatenate(batches)
    return batches

def calculate_anisotropy_svd(X, center=False):
    """
    Calculate anisotropy using SVD as defined in the paper.
    
    Anisotropy is computed as the ratio of the largest squared singular value
    to the sum of all squared singular values of the centered embedding matrix.
    
    Args:
        X: numpy array of shape (num_samples, embedding_dim)
        center: whether to center the embeddings (subtract mean)
    
    Returns:
        anisotropy: scalar value between 0 and 1
                   (higher = more anisotropic)
    """
    # Center the embeddings (subtract mean)
    if center:
        X_centered = X - np.mean(X, axis=0, keepdims=True)
    else:
        X_centered = X
    
    # Perform SVD: X = U @ diag(S) @ V^T
    # We only need singular values
    from scipy.linalg import svd
    U, singular_values, Vt = svd(X_centered, full_matrices=False)
    
    # Compute anisotropy = σ₁² / Σᵢ σᵢ²
    sigma_squared = singular_values ** 2
    anisotropy = sigma_squared[0] / np.sum(sigma_squared)
    
    return anisotropy

def parse_twonn_dimention(layer : int,
                    num_samples : int = 8):
    A = torch.Tensor(parse_layer(layer))[:num_samples]
    dim = twonn_pytorch(A)
    return float(dim)

def parse_anisotropy(layer : int,
                     num_samples : int = 8):
    X = parse_layer(layer)[:num_samples]
    return float(calculate_anisotropy_svd(X))

def parse_singular_dimention(layer : int,
                             num_samples : int,
                             variance_threshold=0.90):
    X = parse_layer(layer)[:num_samples]
    # Вычисляем SVD разложение
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Вычисляем квадраты сингулярных чисел (дисперсии)
    variance = S ** 2
    
    # Вычисляем общую дисперсию
    total_variance = np.sum(variance)
    
    # Вычисляем кумулятивную дисперсию
    cumulative_variance = np.cumsum(variance) / total_variance
    
    # Находим минимальное k
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return float(k)

from tqdm.auto import tqdm
for filename in os.listdir('activations_topics/open_llama_3b'):
    topic = filename

    stats = {
        'twoNN_dim':[],
        'anisotropy':[],
        'singular_dim_09':[]
    }
    print(f'Parse topic {topic}')
    SAVE_STATS_PATH = f'all_stats/openLlama3B_topic_{topic}.yaml'
    DIRNAME = f'activations_topics/open_llama_3b/{topic}'

    for i in tqdm(range(len(os.listdir(DIRNAME)))):
        stats['twoNN_dim'].append(parse_twonn_dimention(i, num_samples=-1))
        stats['anisotropy'].append(parse_anisotropy(i, num_samples=-1))
        stats['singular_dim_09'].append(parse_singular_dimention(i, num_samples=-1, variance_threshold=0.9))
        
    import yaml
    with open(f'{SAVE_STATS_PATH}', 'w') as f:
        yaml.dump(stats, f)