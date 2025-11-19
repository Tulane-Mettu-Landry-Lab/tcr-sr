import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional

class TCRDataset(Dataset):
    def __init__(self, path:str, keys:list=[], mapping:Optional[dict]=None):
        self.path = path
        self.keys = keys
        self.df = pd.read_csv(path)
        if mapping is None:
            self._mapping = {
                'lab.binder': 'labels',
                'tcr.a.cdr3.aa.hidden.states': 'hidden_states_a',
                'tcr.a.cdr3.aa.attention.mask': 'attention_mask_a',
                'tcr.b.cdr3.aa.hidden.states': 'hidden_states_b',
                'tcr.b.cdr3.aa.attention.mask': 'attention_mask_b',
                'epi.aa.hidden.states': 'hidden_states_e',
                'epi.aa.attention.mask': 'attention_mask_e',
            }
        else:
            self._mapping = mapping
    
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        _sample = self.df.iloc[index][self.keys].to_dict()
        _sample = {
            k:torch.tensor(json.loads(v))
            if k.endswith('.enc')
            else v
            for k,v
            in _sample.items()
        }
        return _sample
    
    def collate_fn(self, batch):
        return {self._mapping[k]:batch[k] for k in self._mapping if k in batch}