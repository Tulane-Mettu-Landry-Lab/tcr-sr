import json
import torch
import os
from torch.utils.data import default_collate
from typing import OrderedDict
from itertools import chain

class EmbeddingCollate(object):
    def __init__(self, feature_path:str, keys:list=[], collate_fn=lambda x:x):
        index_path = os.path.join(feature_path, 'index.json')
        with open(index_path, 'r') as f:
            _index_json = json.load(f)
        _index_map = OrderedDict({s:i for i, s in enumerate(chain(*_index_json.values()))})
        self._index_map = _index_map
        self._hidden_states_all = torch.load(os.path.join(feature_path, 'hidden_states.pt')).float()
        self._attention_mask_all = torch.load(os.path.join(feature_path, 'attention_mask.pt')).float()
        self._keys = keys
        self._collate_fn = collate_fn
        self.dtype = self._hidden_states_all.dtype
        self.hidden_dim = self._hidden_states_all.size(-1)
    
    def _fetch_embeddings(self, aa_list:list):
        aa_index = list(map(lambda x:self._index_map[x], aa_list))
        return self._hidden_states_all[aa_index], self._attention_mask_all[aa_index]
    
    def _collate_onekey(self, batch, key:str):
        if key in batch:
            batch[key+'.hidden.states'], batch[key+'.attention.mask'] = self._fetch_embeddings(batch[key])
        return batch

    def collate(self, batch):
        batch = default_collate(batch)
        for key in self._keys:
            batch = self._collate_onekey(batch, key)
        return self._collate_fn(batch)