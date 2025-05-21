import copy

import numpy as np
import torch
from torch.utils import data

from Datasets import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_NORM = 10


def get_acc_loss(loader, model, w_decay):
    """
    Evaluate loss and accuracy of `model` on all batches from `loader`.
    If `w_decay` is provided, adds ½ * w_decay * ||params||^2 to the average loss.
    Returns: (avg_loss, accuracy)
    """
    model.eval().to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0.0
    total_correct = 0
    n_samples = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device).long()
            batch_size = Xb.size(0)

            preds = model(Xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item()

            # accuracy
            pred_labels = preds.argmax(dim=1)
            total_correct += (pred_labels == yb).sum().item()
            n_samples += batch_size

    avg_loss = total_loss / n_samples
    accuracy = total_correct / n_samples

    # optional L2 regularization term
    if w_decay is not None:
        params = get_model_params([model])[0]
        avg_loss += 0.5 * w_decay * float((params**2).sum())

    model.train()
    return avg_loss, accuracy


def set_client_from_params(model, params, strict=True):
    dict_param = copy.deepcopy(dict(model.state_dict()))
    idx = 0
    for name, param in model.state_dict().items():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length
    model.load_state_dict(dict_param, strict=strict)
    return model


def get_model_params(model_list, n_par=None):
    if n_par == None:
        exp_model = model_list[0]
        n_par = 0
        for name, param in exp_model.state_dict().items():
            n_par += len(param.data.reshape(-1))
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, model in enumerate(model_list):
        idx = 0
        for name, param in model.state_dict().items():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)
