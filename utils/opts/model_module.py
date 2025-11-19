import torch
import json
import os
import torch.optim as optim
from typing import Union
from .train import train_model
from .test import test_model
from ..models import ModelLib
from ..data import DataModule

class ModelModule(object):
    def __init__(
        self,
        model_name:str,
        model_config:dict={},
        optimizer_name:str='adamw',
        optimizer_config:dict={},
        training_config:dict={},
        tqdm_bar:bool=True,
        device:str='cpu',
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.optimizer_name = optimizer_name
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.device = device
        self.tqdm_bar = tqdm_bar
        
    def build_model(self, device='cpu', dtype=None, hidden_dim=128):
        _model = ModelLib[self.model_name].from_config(self.model_config, hidden_dim=hidden_dim)
        _model = _model.to(dtype=dtype, device=device)
        return _model
    
    _optimizers = dict(
        adamw = optim.AdamW,
        adam = optim.Adam,
    )
    
    def build_optimizer(self, model, optimizer_name='adamw', configs:dict={}):
        return self._optimizers[optimizer_name](model.parameters(), **configs)
    


    def train(self, traindata:DataModule, resume:bool=False, reg_config=None):
        _model = self.build_model(
            device=self.device,
            dtype=traindata.dtype,
            hidden_dim=traindata.hidden_dim
        )
        if resume:
            if isinstance(resume, str):
                _weight_path = resume
            else:
                _weight_path = os.path.join(self.training_config['save_path'], 'last.pt')
            if os.path.exists(_weight_path):
                _state_dict = torch.load(_weight_path, map_location=self.device)
                _model.load_state_dict(_state_dict, strict=False)
                print(f'[ACT] Resumed from {_weight_path}')
        _optimizer = self.build_optimizer(
            model=_model,
            optimizer_name=self.optimizer_name,
            configs=self.optimizer_config
        )
        if reg_config is not None:
            reg_config = {
                'dataloader': reg_config['dataloader'],
                'device': self.device,
                'epoch': reg_config['epoch'],
                'tqdm_bar': self.tqdm_bar,
                'optimizer': self.build_optimizer(
                                model=_model,
                                optimizer_name=reg_config['optimizer_name'],
                                configs=reg_config['optimizer_config'],
                            )
            }
        train_model(
            model = _model,
            dataloader=traindata.dataloader,
            optimizer=_optimizer,
            device = self.device,
            epoch = self.training_config['epoch'],
            tqdm_bar = self.tqdm_bar,
            save_path = self.training_config['save_path'],
            save_per = self.training_config['save_per'],
            regularize_config=reg_config
        )
        
    def test(self, testdata:DataModule, weights:str='best'):
        _model = self.build_model(
            device=self.device,
            dtype=testdata.dtype,
            hidden_dim=testdata.hidden_dim
        )
        if os.path.isfile(weights):
            _state_dict = torch.load(weights, map_location=self.device)
        else:
            if weights == 'best':
                _weight_path = os.path.join(self.training_config['save_path'], 'best.pt')
            elif weights == 'last':
                _weight_path = os.path.join(self.training_config['save_path'], 'last.pt')
            else:
                _weight_path = os.path.join(self.training_config['save_path'], 'epoch', f'{weights}.pt')
            _state_dict = torch.load(_weight_path, map_location=self.device)
            print(f'[ACT] Resumed from {_weight_path}')
        _state_dict = {k[len("module."):] if k.startswith("module.") else k:v for k,v in _state_dict.items()}
        _model.load_state_dict(_state_dict)
        y_true, y_pred = test_model(
            model=_model,
            dataloader=testdata.dataloader,
            device=self.device
        )
        return y_true, y_pred
                
        
    @classmethod
    def from_config(cls, config:Union[dict,str], **kwargs):
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config, 'r') as f: config = json.load(f)
            else:
                config = json.loads(f)
        config.update(kwargs)
        return cls(**config)