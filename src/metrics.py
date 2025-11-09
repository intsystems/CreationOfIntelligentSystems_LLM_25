from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    confusion_matrix, fbeta_score
)
import numpy as np


def find_optimal_threshold(y_true, y_pred_proba, metric='f1', beta=1.0):
    """
    Подбор оптимального порога по заданной метрике.
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки (0 или 1)
    y_pred_proba : array-like
        Предсказанные вероятности для положительного класса
    metric : str, default='f1'
        Метрика для оптимизации: 'f1', 'f2', 'precision', 'recall', 'accuracy'
    beta : float, default=1.0
        Параметр beta для fbeta_score (используется только если metric='fbeta')
    
    Returns:
    --------
    tuple : (optimal_threshold, best_score, all_thresholds, all_scores)
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'fbeta':
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    best_score = scores[optimal_idx]
    
    return optimal_threshold, best_score, thresholds, scores


def compute_classification_metrics(y_true, y_pred_proba, threshold=None, 
                                   y_val_true=None, y_val_pred_proba=None,
                                   optimize_metric='f1'):
    """
    Вычисление всех метрик классификации с опциональным подбором threshold на валидации.
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки тестового набора (0 или 1)
    y_pred_proba : array-like
        Предсказанные вероятности для тестового набора
    threshold : float or None, default=None
        Порог для преобразования вероятностей в классы.
        Если None и переданы валидационные данные, будет подобран оптимальный.
    y_val_true : array-like or None, default=None
        Истинные метки валидационного набора
    y_val_pred_proba : array-like or None, default=None
        Предсказанные вероятности для валидационного набора
    optimize_metric : str, default='f1'
        Метрика для оптимизации threshold на валидации
    
    Returns:
    --------
    dict : Словарь со всеми метриками
    """
    # Подбор threshold на валидации, если данные переданы
    if threshold is None and y_val_true is not None and y_val_pred_proba is not None:
        threshold, val_score, thresholds, scores = find_optimal_threshold(
            y_val_true, y_val_pred_proba, metric=optimize_metric
        )
    elif threshold is None:
        threshold = 0.5
    
    # Преобразование вероятностей в бинарные предсказания
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Компоненты confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Вычисление всех метрик
    metrics = {
        # Метрики на основе порога
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        
        # Компоненты confusion matrix
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        
        # Дополнительные метрики
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        
        # Метрики на основе вероятностей (не зависят от порога)
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        
        # Использованный порог
        'threshold': threshold
    }
    for key, value in metrics.items():
        if not isinstance(value, int):
            metrics[key] = float(value)
    return metrics