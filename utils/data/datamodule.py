from .tcr_dataset import TCRDataset
from .embedding_collate import EmbeddingCollate
from torch.utils.data import DataLoader
from typing import Union, Optional

import os
import json

class DataModule(object):
    
    def __init__(
        self,
        data_path:str,
        embedding_path:str,
        embedding_keys:list[str],
        concept_keys:list[str],
        label_keys:list[str],
        mapping:dict,
        batch_size:int=512,
        shuffle:bool=False,
        num_workers:int=8,
        dataset_class=TCRDataset
    ) -> None:
        self.data_path = data_path
        self.embedding_path = embedding_path
        self.embedding_keys = embedding_keys
        self.concept_keys = concept_keys
        self.label_keys = label_keys
        self.mapping = mapping
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.keys = embedding_keys + concept_keys + label_keys
        
        self.dataset = dataset_class(
            path = self.data_path,
            keys = self.keys,
            mapping = self.mapping
        )
        
        self.collator = EmbeddingCollate(
            feature_path=self.embedding_path,
            keys=self.embedding_keys,
            collate_fn=self.dataset.collate_fn
        )
        
        self.dtype = self.collator.dtype
        self.hidden_dim = self.collator.hidden_dim
        
        self.dataloader = DataLoader(
            dataset = self.dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            collate_fn = self.collator.collate,
            num_workers = self.num_workers
        )
        
    def __len__(self):
        return len(self.dataset)
    
    @classmethod
    def from_config(cls, config:Union[dict,str], **kwargs):
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config, 'r') as f: config = json.load(f)
            else:
                config = json.loads(f)
        return cls(**config, **kwargs)
    
    def to_config(self, path:Optional[str]=None):
        _config = dict(
            data_path = self.data_path,
            embedding_path = self.embedding_path,
            embedding_keys = self.embedding_keys,
            concept_keys = self.concept_keys,
            label_keys = self.label_keys,
            mapping = self.mapping,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
        )
        if path is None: return _config
        else:
            with open(path, 'w') as f:
                json.dump(_config, f, indent=2)