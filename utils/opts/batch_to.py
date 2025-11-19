def batch_to(batch, device=None, dtype=None):
    if dtype is not None:
        batch = {k:v.to(dtype=dtype) for k,v in batch.items()}
    if device is not None:
        batch = {k:v.to(device=device) for k,v in batch.items()}
    return batch