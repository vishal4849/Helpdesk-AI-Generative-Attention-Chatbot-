import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    """
    Attention Mechanism supporting both Bahdanau (Additive) and Luong (Multiplicative) styles.
    
    Bahdanau score: v_a^T * tanh(W_a * dec_hidden + U_a * enc_outputs)
    Luong General:  dec_hidden^T * W_a * enc_outputs
    """
    def __init__(
        self,
        enc_hid_dim: int,
        dec_hid_dim: int,
        attention_type: str = "bahdanau",
        bidirectional_encoder: bool = True
    ):
        super().__init__()
        self.attention_type = attention_type.lower()
        self.enc_total_dim = enc_hid_dim * (2 if bidirectional_encoder else 1)
        self.dec_hid_dim = dec_hid_dim

        if self.attention_type == "bahdanau":
            # Additive Attention
            self.W_dec = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
            self.W_enc = nn.Linear(self.enc_total_dim, dec_hid_dim, bias=False)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif self.attention_type == "luong":
            # Multiplicative (General) Attention
            self.W_general = nn.Linear(self.enc_total_dim, dec_hid_dim, bias=False)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Choose 'bahdanau' or 'luong'.")

    def forward(
        self,
        dec_hidden: torch.Tensor,
        enc_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            dec_hidden: [batch_size, dec_hid_dim] (current decoder hidden state)
            enc_outputs: [batch_size, src_len, enc_total_dim] (all encoder outputs)
            mask: [batch_size, src_len] boolean or binary mask (1 for valid token, 0 for PAD)
        Returns:
            attention_weights: [batch_size, src_len] normalized attention probabilities
        """
        src_len = enc_outputs.shape[1]

        if self.attention_type == "bahdanau":
            # Repeat decoder hidden state for each source time step
            # dec_hidden: [batch_size, dec_hid_dim] -> [batch_size, src_len, dec_hid_dim]
            dec_rep = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

            # Energy = v^T * tanh(W_dec * dec_hidden + W_enc * enc_outputs)
            energy = torch.tanh(self.W_dec(dec_rep) + self.W_enc(enc_outputs))  # [batch_size, src_len, dec_hid_dim]
            attention_scores = self.v(energy).squeeze(2)  # [batch_size, src_len]

        elif self.attention_type == "luong":
            # score = dec_hidden^T * W * enc_output
            # projected_enc: [batch_size, src_len, dec_hid_dim]
            projected_enc = self.W_general(enc_outputs)
            # dec_hidden: [batch_size, dec_hid_dim, 1]
            dec_hidden_exp = dec_hidden.unsqueeze(2)
            # bmm: [batch_size, src_len, dec_hid_dim] x [batch_size, dec_hid_dim, 1] -> [batch_size, src_len, 1]
            attention_scores = torch.bmm(projected_enc, dec_hidden_exp).squeeze(2)

        # Apply masking if provided
        if mask is not None:
            # Mask out PAD tokens by setting scores to a large negative number
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        # Softmax over source sequence length
        return F.softmax(attention_scores, dim=1)
