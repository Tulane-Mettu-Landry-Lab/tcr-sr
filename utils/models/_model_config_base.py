import os
import json
import torch.nn as nn
from typing import Union

class ConfigBase(nn.Module):
    
    @classmethod
    def from_config(cls, config:Union[dict,str], **kwargs):
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config, 'r') as f: config = json.load(f)
            else:
                config = json.loads(f)
        config.update(kwargs)
        return cls(**config)