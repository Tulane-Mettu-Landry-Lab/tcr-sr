import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional

class TCRXAIDataset(Dataset):
    def __init__(self, path:str, keys:list=[], mapping:Optional[dict]=None):
        self.path = path
        self.keys = keys
        with open(path, 'r') as f:
            self.data = json.load(f)
        if mapping is None:
            self._mapping = {
                'lab.binder': 'labels',
                'tcr.a.cdr3.aa.hidden.states': 'hidden_states_a',
                'tcr.a.cdr3.aa.attention.mask': 'attention_mask_a',
                'tcr.b.cdr3.aa.hidden.states': 'hidden_states_b',
                'tcr.b.cdr3.aa.attention.mask': 'attention_mask_b',
                'epi.aa.hidden.states': 'hidden_states_e',
                'epi.aa.attention.mask': 'attention_mask_e',
                'lab.cm.epi.cdr3a': 'labels_contact_map_ae',
                'lab.cm.epi.cdr3b': 'labels_contact_map_be',
                'lab.resolution': 'labels_contact_map_ae_quality',
                'lab.resolution': 'labels_contact_map_be_quality',
            }
        else:
            self._mapping = mapping
    
    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        _sample = self.data[index]
        _sample = {k:_sample[k] for k in self.keys}
        _sample = {
            k:torch.tensor(v)
            if k[:6] == 'lab.cm'
            else v
            for k,v
            in _sample.items()
        }
        return _sample
    
    def collate_fn(self, batch):
        return {self._mapping[k]:batch[k] for k in self._mapping if k in batch}