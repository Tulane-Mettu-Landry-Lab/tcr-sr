import torch
import torch.nn as nn
from .cmap_proto import ContactMapProto

class ContactAreaProto(nn.Module):
    def __init__(
        self,
        sample_distances=torch.arange(0.5, 1.0, 0.1),
        hidden_dim=1280,
        num_heads=1,
        dropout=0.1
    ):
        super().__init__()
        self.cmp = ContactMapProto(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.register_buffer('sample_distances', torch.tensor(sample_distances))
        self.register_buffer('areas', torch.softmax(self.sample_distances, dim=0))
        self.contact_map = None
        
    def forward(
        self,
        hidden_states_a,
        attention_mask_a,
        hidden_states_b,
        attention_mask_b,
    ):
        _total_area = (attention_mask_a[:,:,None]*attention_mask_b[:,None,:]).sum(dim=[-1,-2])
        _contact_map = self.cmp(
            hidden_states_a,
            attention_mask_a,
            hidden_states_b,
            attention_mask_b,
        )
        self.contact_map = _contact_map
        _occ = torch.sigmoid((_contact_map[:, None, :, :] - self.sample_distances[None,:,None,None])*50)
        _occ = (_occ.sum(dim=[-1,-2])/_total_area[:, None])**0.5
        _occ = (_occ*self.areas[None, :]).sum(dim=-1)
        return _occ