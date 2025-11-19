import torch
import torch.nn as nn
from .coatten import CoAttentionLayer
from .proto_linear import ProtoLinear
class AlleleConcepts(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads=1,
        dropout=0.1,
        select_threshold=0.9,
        concept_nums={
            'a_v': 54,
            'a_j': 66,
            'b_v': 44,
            'b_j': 21,
            'tcr_species': 9,
            'mhc_allele': 172,
            'mhc_class': 4
        }
    ):
        super().__init__()
        # self.coatten = CoAttentionLayer(hidden_size=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.concept_a_v = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['a_v'], dropout=dropout, select_threshold=select_threshold)
        self.concept_a_j = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['a_j'], dropout=dropout, select_threshold=select_threshold)
        self.concept_b_v = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['b_v'], dropout=dropout, select_threshold=select_threshold)
        self.concept_b_j = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['b_j'], dropout=dropout, select_threshold=select_threshold)
        self.concept_tcr_species_a = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['tcr_species'], dropout=dropout, select_threshold=select_threshold)
        self.concept_tcr_species_b = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['tcr_species'], dropout=dropout, select_threshold=select_threshold)
        self.concept_mhc_allele = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['mhc_allele'], dropout=dropout, select_threshold=select_threshold)
        self.concept_mhc_class = ProtoLinear(hidden_dim=hidden_dim, proto_num=concept_nums['mhc_class'], dropout=dropout, select_threshold=select_threshold)
        
        self.loss = None

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
        
        concept_weights = [
            self.concept_a_v(hidden_states_a, attention_mask_a, tcr_a_v_concepts),
            self.concept_a_j(hidden_states_a, attention_mask_a, tcr_a_j_concepts),
            self.concept_b_v(hidden_states_b, attention_mask_b, tcr_b_v_concepts),
            self.concept_b_j(hidden_states_b, attention_mask_b, tcr_b_j_concepts),
            self.concept_tcr_species_a(hidden_states_a, attention_mask_a, tcr_species_concepts),
            self.concept_tcr_species_b(hidden_states_b, attention_mask_b, tcr_species_concepts),
            self.concept_mhc_allele(hidden_states_e, attention_mask_e, mhc_allele_concepts),
            self.concept_mhc_class(hidden_states_e, attention_mask_e, mhc_class_concepts),
        ]
        
        
        if self.training:
            self.loss = torch.stack([
                self.concept_a_v.loss,
                self.concept_a_j.loss,
                self.concept_b_v.loss,
                self.concept_b_j.loss,
                self.concept_tcr_species_a.loss,
                self.concept_tcr_species_b.loss,
                self.concept_mhc_allele.loss,
                self.concept_mhc_class.loss
            ]).mean()

        return torch.mean(torch.stack(concept_weights), dim=0)