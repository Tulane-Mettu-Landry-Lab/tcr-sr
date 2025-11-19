import torch
import torch.nn as nn
import torch.nn.functional as F
from .coatten import CoAttentionLayer


class ContactMapProto(nn.Module):
    def __init__(self, hidden_dim=1280, num_heads=1, dropout=0.1):
        super().__init__()
        self.coatten = CoAttentionLayer(hidden_size=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.amp_factor = nn.Parameter(torch.tensor([2], dtype=float))
    
    def feature_distance_matrix(self, a, b, a_mask, b_mask):
        p_as = F.normalize(a, dim=-1)
        p_bs = F.normalize(b, dim=-1)
        s = (p_as @ p_bs.permute([0, 2, 1]))
        s *= self.amp_factor
        s = torch.sigmoid(s)
        s = s*a_mask[:,:,None]*b_mask[:,None,:]
        return s
    
    def forward(
        self,
        hidden_states_a,
        attention_mask_a,
        hidden_states_b,
        attention_mask_b,
    ):
        co_a, co_b = self.coatten(
            hidden_states_a,
            hidden_states_b,
            attention_mask_a,
            attention_mask_b,
        )
        _contact_map = self.feature_distance_matrix(
            co_a, co_b,
            attention_mask_a,
            attention_mask_b
        )
        return _contact_map