import torch
import logging
import numpy as np


def log_print(msg):
    """Log printing function

    Parameters
    -----------
    msg: 
        The information to be printed
    """

    logging.info(msg)
    print(msg)


def top_k(label_pred, labels, k):
    """Calculate the top_k accuracy

    Parameters
    -----------
    label_pred: 
        Predicted labels
    labels: 
        Truth labels
    k: 
        Top_(k) accuracy
    """

    pred_topk = torch.topk(label_pred, k=k, dim=1)[1].detach().cpu().numpy()
    true_label = labels.detach().cpu().numpy()
    expanded_true_label = np.tile(true_label.reshape(-1, 1), (1, k))
    correct_predictions = np.sum(np.equal(pred_topk, expanded_true_label), axis=1)
    topk_acc = np.mean(correct_predictions)
    
    return topk_acc