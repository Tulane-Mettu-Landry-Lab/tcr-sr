import torch
import torch.nn as nn
from ._model_config_base import ConfigBase
from ._lib import ModelLib

@ModelLib.register
class LinearClassifier(ConfigBase):
    def __init__(
        self,
        hidden_dim: int,
        class_num: int = 2,
        global_model:str = 'first',
        dropout:float = 0.25
    ):
        super().__init__()
        # Input = 3 * hidden_dim, output = 1 (binary prediction)
        self.global_model = global_model
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden_dim, 3 * hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(3 * hidden_dim, class_num, bias=False),
        )
        # self.fc = nn.Linear(3 * hidden_dim, class_num, dtype=dtype)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2175, 0.7825]))
        
        self.loss = None

    def _get_global_feature(self, hidden_states, attention_masks):
        if self.global_model == 'first':
            return hidden_states[:, 0, :]
        else:
            return hidden_states.sum(dim=1) / attention_masks.sum(dim=1, keepdims=True)

    def forward(
        self,
        hidden_states_a, attention_mask_a,
        hidden_states_b, attention_mask_b,
        hidden_states_e, attention_mask_e,
        mhc_allele_concepts, mhc_class_concepts,
        tcr_a_v_concepts, tcr_a_j_concepts,
        tcr_b_v_concepts, tcr_b_j_concepts,
        tcr_species_concepts,
        labels,
    ):
        # Take the [CLS] token (first token)
        a_cls = self._get_global_feature(hidden_states_a, attention_mask_a)
        b_cls = self._get_global_feature(hidden_states_b, attention_mask_b)
        e_cls = self._get_global_feature(hidden_states_e, attention_mask_e)
        # Concatenate
        concat = torch.cat([a_cls, b_cls, e_cls], dim=-1)  # [batch, 3*dim]

        # Linear layer
        outputs = self.fc(concat)  # [batch, 1]
        
        if self.training:
            _loss = self.criterion(outputs, labels.to(dtype=torch.long))
            self.loss = _loss
            _loss.backward()

        return outputs