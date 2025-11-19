import os
import torch
import json
from .f_epoch import forward_epoch

@torch.no_grad
def test_model(
    model,
    dataloader,
    device='cpu',
):
    model = model.eval()
    with torch.no_grad():
        y_true, y_pred = \
            forward_epoch(
                model,
                dataloader,
                device=device,
            )
    return y_true, y_pred