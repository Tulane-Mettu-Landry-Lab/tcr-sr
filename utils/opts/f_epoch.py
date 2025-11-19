import torch
import numpy as np
from tqdm import tqdm
from .f_iter import forward_iter

def forward_epoch(model, dataloader, device='cpu', optimizer=None, tqdm_bar=True, epoch=0, customize_mode=None):
    _mode = 'train' if model.training else 'eval'
    if customize_mode is not None:
        _mode = customize_mode
    if tqdm_bar:
        bar = tqdm(dataloader, desc=f'[{_mode}; EPOCH={epoch}]')
    else:
        bar = dataloader
    loss_total, acc_total, positive_total, total = 0, 0, 0, 0
    y_pred, y_true = None, None
    for batch in bar:
        outputs, labels, loss, _batch_size = \
            forward_iter(
                model = model,
                batch = batch,
                device = device,
                optimizer = optimizer
            )
        if model.training:
            total += _batch_size
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            labels = labels.cpu()
            acc_total += (predicted == labels).sum().item()
            positive_total += predicted.sum().item()
            loss_total += loss * _batch_size
            if tqdm_bar: bar.set_description(f'[{_mode}; EPOCH={epoch}] LOSS={loss_total/total:.6f} ACC={acc_total/total:.3f} POSR={positive_total/total:.3f}')
        else:
            if y_pred is None:
                y_pred = outputs.cpu().detach().numpy()
            else:
                y_pred = np.concatenate([y_pred, outputs.cpu().detach().numpy()])
            if y_true is None:
                y_true = labels.cpu().detach().numpy()
            else:
                y_true = np.concatenate([y_true, labels.cpu().detach().numpy()])
    
    if model.training:
        return loss_total/total, acc_total/total, positive_total/total
    else:
        return y_true, y_pred