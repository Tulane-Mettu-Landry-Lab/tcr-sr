import torch
import torch.nn as nn
from .cmap_proto import ContactMapProto

class ContactAreaProtoRegularize(nn.Module):
    def __init__(
        self,
        sample_distances=torch.arange(0.5, 1.0, 0.1),
        hidden_dim=1280,
        num_heads=1,
        dropout=0.1,
        regular_top_k=0.75,
    ):
        super().__init__()
        self.cmp = ContactMapProto(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.register_buffer('sample_distances', torch.tensor(sample_distances))
        self.register_buffer('areas', torch.softmax(self.sample_distances, dim=0))
        self.regular_top_k = regular_top_k
        self.contact_map = None
        self.loss = None
        
    def _norm_contact_map_labels(
        self,
        contact_map_labels:torch.tensor,
    ):
        contact_map_labels = torch.where(contact_map_labels > 0, contact_map_labels, torch.max(contact_map_labels))
        _min = contact_map_labels.min(dim=1).values.min(dim=1).values[:, None, None]
        _max = contact_map_labels.max(dim=1).values.max(dim=1).values[:, None, None]
        contact_map_labels = (contact_map_labels - _min) / (_max - _min)
        contact_map_labels = 1-contact_map_labels
            
        return contact_map_labels
    
    def _proj_scale(
        self,
        contact_map_labels:torch.tensor,
        scale_max:torch.tensor,
        scale_min:torch.tensor
    ):
        scale_max = scale_max[:, None, None]
        scale_min = scale_min[:, None, None]
        _scale = scale_max - scale_min
        contact_map_labels = contact_map_labels * _scale + scale_min
        
        return contact_map_labels
    
    def _norm_resolution(
        self,
        resolutions:torch.tensor,
    ):
        _exp_res = torch.exp(-resolutions)
        return _exp_res / _exp_res.sum()
        
    def forward(
        self,
        hidden_states_a,
        attention_mask_a,
        hidden_states_b,
        attention_mask_b,
        labels_contact_map=None,
        labels_contact_map_quality=None,
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
        if labels_contact_map is not None and self.training:
            _normed_labels_contact_map = self._norm_contact_map_labels(
                labels_contact_map,
            )
            _mask = (_normed_labels_contact_map >= self.regular_top_k)
            _normed_labels_contact_map = self._proj_scale(
                _normed_labels_contact_map,
                scale_min=_contact_map.min(dim=1).values.min(dim=1).values,
                scale_max=_contact_map.max(dim=1).values.max(dim=1).values,
            )
            _full_mse = nn.functional.mse_loss(_contact_map, _normed_labels_contact_map, reduction='none')
            _full_cos = 1 - (nn.functional.normalize(_contact_map, dim=-1) @ nn.functional.normalize(_normed_labels_contact_map, dim=-1)).transpose(-1, -2)
            
            _masked_mse = torch.where(_mask, _full_mse, torch.nan)
            # _masked_cos = torch.where(_mask, _full_cos, torch.nan)
            _nmasked_region = torch.where(~_mask, _contact_map, torch.nan)
            _weighted_masked_mse = _masked_mse.nanmean(dim=[1, 2])
            _weighted_masked_cos = _full_cos.mean(dim=[1, 2])
            # _weighted_nmasked_region = _nmasked_region.nanmean(dim=[1, 2])
            # _weighted_masked_loss = (_weighted_masked_mse + _weighted_masked_cos) / 2
            _weighted_masked_loss = _weighted_masked_mse
            _weighted_masked_loss *= self._norm_resolution(labels_contact_map_quality)
            self.loss = (_weighted_masked_loss).sum()
            
        return _occ