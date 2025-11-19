import torch
from .batch_to import batch_to
def forward_iter(model, batch, device='cpu', optimizer=None):
    if model.training and optimizer:
        optimizer.zero_grad()
    batch = batch_to(batch, device=device)
    outputs = model(**batch)
    loss = None
    if isinstance(outputs, tuple):
        outputs, loss = outputs
    if model.training:
        if loss is None:
            loss = model.loss
        # loss = loss.mean() # for data parallel!
        loss.backward()
        loss = loss.item()
        if optimizer:
            if isinstance(model, torch.nn.DataParallel):
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    else:
        loss = None
    labels = batch['labels']
    _batch_size = labels.size(0)
    return outputs, labels, loss, _batch_size
