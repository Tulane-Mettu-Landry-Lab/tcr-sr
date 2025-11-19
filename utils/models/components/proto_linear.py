import torch
import torch.nn as nn
class ProtoLinear(nn.Module):
    
    def __init__(
        self,
        hidden_dim=1280,
        proto_num=512,
        dropout=0.2,
        select_threshold=0.75,
    ):
        super().__init__()
        self.select_threshold = select_threshold
        self.proto_num = proto_num
        self.prelinear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.protolinear = nn.Sequential(
            # nn.Conv1d(hidden_dim, proto_num, 1),
            nn.Linear(hidden_dim, proto_num),
            nn.Sigmoid(),
        )
        self.protosel = nn.Sequential(
            nn.Linear(hidden_dim, proto_num),
            # nn.Conv1d(hidden_dim, proto_num, 1),
            nn.Sigmoid(),
        )
        self.proto_weights = None
        self.proto_selects = None
        self.weights = None
        self.loss = None
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.amp_factor = nn.Parameter(torch.tensor([2], dtype=float))
        
    def forward(self, hidden_states, attention_mask, concepts=None):
        _s_batch, _s_token, _s_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, _s_dim)
        _mapped_hidden_states = self.prelinear(hidden_states)
        _mapped_hidden_states = _mapped_hidden_states.reshape(*_mapped_hidden_states.shape, 1)
        # print(_mapped_hidden_states.shape)
        _proto_weights = self.protolinear(_mapped_hidden_states.flatten(start_dim=-2, end_dim=-1))
        _proto_selects = self.protosel(_mapped_hidden_states.flatten(start_dim=-2, end_dim=-1))
        # _proto_weights = self.protolinear(_mapped_hidden_states)
        # _proto_selects = self.protosel(_mapped_hidden_states)
        # print(_proto_weights.shape)
        
        
        if self.select_threshold is None:
            _ids = _proto_selects.argmax(dim=-2)
            _mask = nn.functional.one_hot(_ids, self.proto_num)
            _mask = _mask.reshape(_mask.size(0), -1)
        else:
            _mask = torch.sigmoid((_proto_selects-self.select_threshold)*self.proto_num**0.5)
            # _mask = _proto_selects
            self.proto_selects = _mask.reshape(_s_batch, _s_token, -1)
            _mask = _mask.reshape(_mask.size(0), -1)
        _proto_weights = _proto_weights.reshape(_proto_weights.size(0), -1)
        # _proto_selects = _proto_selects.reshape(_s_batch, _s_token, -1)
        _weights = ((_proto_weights * _mask).sum(dim=-1) / _mask.sum(dim=-1))
        _weights = _weights.reshape(_s_batch, _s_token, 1)
        _proto_weights = _proto_weights.reshape(_s_batch, _s_token, -1)
        self.proto_weights = _proto_weights
        _mask = _mask.reshape(_s_batch, _s_token, -1)
        self.weights = _weights
        
        if self.training:
            outputs = _mask.mean(dim=1)
            multi_hot = torch.zeros(*outputs.shape, device=outputs.device)
            multi_hot = multi_hot.scatter_(1, concepts+1, 1)
            self.loss = self.criterion(outputs, multi_hot)
            
        _agg_weights = (attention_mask.float() * _weights[:,:,0]).sum(dim=1) * self.amp_factor
        _avg_weights = _agg_weights / attention_mask.float().sum(dim=1)
        return _avg_weights
        