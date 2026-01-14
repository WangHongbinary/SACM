import torch
import logging
import numpy as np
import torch.nn.functional as F

def log_print(msg):
    """Log printing function

    Parameters
    -----------
    msg: 
        The information to be printed
    """

    logging.info(msg)
    print(msg)


def segment_level_eval(outputs, wav_features, label, batchsize, loss_f, p_k, p_1, topk, loss, temperature, device):
    """Evaluation function to compute loss and assess model performance

    Parameters
    -----------
    outputs: 
        SEEG model output results
    wav_features: 
        Audio features
    label: 
        Ground truth label sequence
    batchsize: 
        The batch size
    loss_f: 
        Type of loss function
    p_k: 
        Number of samples correctly predicted in the top-5
    p_1: 
        Number of samples correctly predicted in the top-1
    topk:
        Top-k accuracy
    loss: 
        Current total loss value
    temperature: 
        Temperature coefficient
    device: 
        CPU or GPU ID
    """


    if loss_f == 'CLIP':
        clip_loss, inner_product = CLIP_Xent(outputs, wav_features, batchsize, temperature, device)
    elif loss_f == 'MSE':
        mse_loss = MSE_loss(outputs, wav_features, batchsize)

    batch_pred = F.softmax(inner_product, dim=1)
    _, batch_pred_index = torch.sort(batch_pred, descending=True)
    pred_label = label[batch_pred_index[:, 0]]

    for i in range(batchsize):
        label = np.array(label)
        pos_index = np.where(label==label[i])[0]
        pos_index = torch.from_numpy(pos_index).to(device)
        if torch.any(torch.isin(batch_pred_index[i][0:int(topk)], pos_index)):
            p_k = p_k + 1
        if torch.any(torch.isin(batch_pred_index[i][0:1], pos_index)):
            p_1 = p_1 + 1

    if loss_f == 'CLIP':
        loss = loss + clip_loss
        return p_k, p_1, loss, clip_loss, pred_label
    
    elif loss_f == 'MSE':
        loss = loss + mse_loss
        return p_k, p_1, loss, mse_loss


def MSE_loss(outputs, wav_features, batchsize):
    """MSE Loss Calculation Function

    Parameters
    -----------
    outputs: 
        SEEG model output results
    wav_features: 
        Audio features
    batchsize: 
        The batch size
    """

    out_min = torch.min(outputs.reshape(batchsize, -1), dim=1).values
    out_max = torch.max(outputs.reshape(batchsize, -1), dim=1).values
    out_min = out_min.reshape(batchsize, 1, 1)
    out_max = out_max.reshape(batchsize, 1, 1)
    out = (outputs - out_min) / (out_max - out_min)
    wav_min = torch.min(wav_features.reshape(batchsize, -1), dim=1).values
    wav_max = torch.max(wav_features.reshape(batchsize, -1), dim=1).values
    wav_min = wav_min.reshape(batchsize, 1, 1)
    wav_max = wav_max.reshape(batchsize, 1, 1)
    wav = (wav_features - wav_min) / (wav_max - wav_min)
    mse_loss = F.mse_loss(out, wav)
    return mse_loss


def CLIP_Xent(outputs, wav_features, batchsize, temperature, device):
    """CLIP Loss Calculation Function

    Parameters
    -----------
    outputs: 
        SEEG model output results
    wav_features: 
        Audio features
    batchsize: 
        The batch size
    temperature: 
        Temperature coefficient
    device: 
        CPU or GPU ID
    """

    outputs = F.normalize(outputs.reshape(batchsize, -1))
    wav_features = F.normalize(wav_features.reshape(batchsize, -1))
    inner_product = torch.mm(outputs, wav_features.T)
    inner_product = temperature * inner_product
    clip_Xent1 = F.cross_entropy(inner_product, torch.tensor(range(batchsize)).to(device))
    clip_Xent2 = F.cross_entropy(inner_product.T, torch.tensor(range(batchsize)).to(device))
    clip_Xent = (clip_Xent1 + clip_Xent2)/2
    return clip_Xent, inner_product


def cal_acc(batch_num, p_k, p_1, loss, batchsize):
    """Calculate the Top-5 and Top-1 accuracy for the entire epoch

    Parameters
    -----------
    batch_num: 
        The number of batches in each epoch
    p_k: 
        The number of samples correctly predicted in Top-5 for each epoch
    p_1: 
        The number of samples correctly predicted in Top-1 for each epoch
    loss: 
        The total loss for the entire epoch
    batchsize: 
        The batch size
    """

    len_data = batch_num * batchsize 
    acc5 = 100 * p_k / len_data
    acc1 = 100 * p_1 / len_data
    loss = loss / batch_num

    return acc5, acc1, loss

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