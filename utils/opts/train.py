import os
import torch
import json
import torch.nn as nn
from .f_epoch import forward_epoch

FORCE_CPU = False # for data parallel!
def regularize_model(
    model,
    dataloader,
    optimizer=None,
    device='cpu',
    epoch=5,
    tqdm_bar=True,
):
    losses = []
    for e in range(epoch):
        loss, acc, pos = \
            forward_epoch(
                model,
                dataloader,
                device=device if not FORCE_CPU else 'cpu',
                optimizer=optimizer,
                tqdm_bar=tqdm_bar,
                epoch=e,
                customize_mode='finetune'
            )
        losses.append(loss)
    return sum(losses) / len(losses)

def train_model(
    model,
    dataloader,
    optimizer=None,
    device='cpu',
    epoch=300,
    tqdm_bar=True,
    save_path='model_ckpts',
    save_per=None,
    regularize_config=None,
):
    os.makedirs(save_path, exist_ok=True)
    if save_per: os.makedirs(os.path.join(save_path, 'epoch'), exist_ok=True)
    losses, accs, poses, reglosses = [], [], [], []
    best_acc = -1
    # model = nn.DataParallel(model).cuda() # for data parallel!
    model = model.train()
    for e in range(epoch):
        if regularize_config is not None:
            reg_loss = regularize_model(model, **regularize_config)
            reglosses.append(reg_loss)
        loss, acc, pos = \
            forward_epoch(
                model,
                dataloader,
                device=device if not FORCE_CPU else 'cpu',
                optimizer=optimizer,
                tqdm_bar=tqdm_bar,
                epoch=e
            )
        
        losses.append(loss)
        accs.append(acc)
        poses.append(pos)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pt'))
        torch.save(model.state_dict(), os.path.join(save_path, 'last.pt'))
        if save_per is not None and e != 0 and e % save_per == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'epoch', f'{e}.pt'))
        
    log = [
        {'epoch': i, 'loss': loss, 'accuray': acc, 'positive': pos, 'regloss': regloss}
        for i, (loss, acc, pos, regloss) in enumerate(zip(losses, accs, poses, reglosses))
    ]
    with open(os.path.join(save_path, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    return losses, accs, poses
