import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention, BertConfig


class CoAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            is_decoder=False,
        )
        # Attention in both directions
        self.a_to_b_attn = BertAttention(config)
        self.b_to_a_attn = BertAttention(config)

    def forward(self, a_hidden_states, b_hidden_states, a_attention_mask=None, b_attention_mask=None):
        """
        a_hidden_states: (batch, len_a, hidden)
        b_hidden_states: (batch, len_b, hidden)
        a_attention_mask: (batch, 1, 1, len_a)
        b_attention_mask: (batch, 1, 1, len_b)
        """
        a_attention_mask = a_attention_mask[:,None, None,:]
        b_attention_mask = b_attention_mask[:,None, None,:]
        # a attends to b (queries from a, keys/values from b)
        a2b_outputs = self.a_to_b_attn(
            hidden_states=a_hidden_states,
            attention_mask=a_attention_mask,
            encoder_hidden_states=b_hidden_states,
            encoder_attention_mask=b_attention_mask,
        )
        a2b_out = a2b_outputs[0]

        # b attends to a (queries from b, keys/values from a)
        b2a_outputs = self.b_to_a_attn(
            hidden_states=b_hidden_states,
            attention_mask=b_attention_mask,
            encoder_hidden_states=a_hidden_states,
            encoder_attention_mask=a_attention_mask,
        )
        b2a_out = b2a_outputs[0]

        return a2b_out, b2a_out