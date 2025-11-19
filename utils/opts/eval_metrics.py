import json
import os
from typing import Union

class EvalMetrics(object):
    
    _metrics = {}
    
    @classmethod
    def register(cls, name):
        def _register(func):
            cls._metrics[name] = func
            return func
        return _register
    
    def __init__(self, **kwargs):
        self.metrics = kwargs
    
    def __len__(self):
        return len(self.metrics)
    
    def eval(self, y_true, y_pred, datamodule):
        return {
            name:self._metrics[mname](y_true, y_pred, datamodule, configs=config)
            for name, (mname, config) in self.metrics.items()
        }
        
    @classmethod
    def from_config(cls, config:Union[dict,str], **kwargs):
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config, 'r') as f: config = json.load(f)
            else:
                config = json.loads(f)
        config.update(kwargs)
        return cls(**config)