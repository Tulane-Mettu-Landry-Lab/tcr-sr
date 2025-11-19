import torch
import json
import torch.nn as nn
from .components import (
    ContactAreaProto, CoAttentionLayer, ContactAreaProtoRegularize
)
from ._model_config_base import ConfigBase
from ._lib import ModelLib

@ModelLib.register
class ContactAreaRegularizeClassifier(ConfigBase):
    def __init__(
        self,
        hidden_dim: int,
        sample_distances=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        num_heads=1,
        dropout=0.1,
        chain_weights=[0.5,0.5],
        label_balance=[0.2175, 0.7825],
        regular_top_k=0.75,
        regular_with_binding=False,
    ):
        super().__init__()
        self.ae_cap = ContactAreaProtoRegularize(
            sample_distances=sample_distances,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            regular_top_k=regular_top_k,
        )
        self.be_cap = ContactAreaProtoRegularize(
            sample_distances=sample_distances,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            regular_top_k=regular_top_k,
        )
        self.coatten = CoAttentionLayer(hidden_size=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.chain_weights = chain_weights
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(label_balance))
        # self.criterion = nn.CrossEntropyLoss()
        
        self.loss = None
        self.regular_with_binding = regular_with_binding

    def forward(
        self,
        hidden_states_a, attention_mask_a,
        hidden_states_b, attention_mask_b,
        hidden_states_e, attention_mask_e,
        labels,
        labels_contact_map_ae=None,
        labels_contact_map_be=None,
        lab_resolution=None,
    ):
        co_a, co_b = self.coatten(
            hidden_states_a,
            hidden_states_b,
            attention_mask_a,
            attention_mask_b,
        )
        w_ae = self.ae_cap(
            co_a,
            attention_mask_a,
            hidden_states_e,
            attention_mask_e,
            labels_contact_map_ae,
            lab_resolution
        )
        w_be = self.be_cap(
            co_b,
            attention_mask_b,
            hidden_states_e,
            attention_mask_e,
            labels_contact_map_be,
            lab_resolution
        )
        if self.chain_weights == 'max':
            w = torch.stack([w_ae, w_be]).max(dim=0).values
        else:
            w_ae *= self.chain_weights[0]
            w_be *= self.chain_weights[1]
            w = w_ae+w_be
        
        
        outputs = torch.stack([1-w, w], dim=1)
        if self.training:
            if labels_contact_map_ae is None or labels_contact_map_be is None:
                _loss = self.criterion(outputs, labels.to(dtype=torch.long))
            else:
                _loss = self.ae_cap.loss
                _loss += self.be_cap.loss
                if self.regular_with_binding:
                    _loss += self.criterion(outputs, labels.to(dtype=torch.long))
                    
            self.loss = _loss

        return outputs, self.loss