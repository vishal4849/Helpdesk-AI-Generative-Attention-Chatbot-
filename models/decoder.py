import torch
import torch.nn as nn
from typing import Tuple, Optional
from models.attention import Attention


class Decoder(nn.Module):
    """
    Attention-guided Decoder module for Seq2Seq architecture.
    """
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        attention: Attention,
        n_layers: int = 1,
        dropout: float = 0.2,
        bidirectional_encoder: bool = True
    ):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = attention
        self.n_layers = n_layers
        self.enc_total_dim = enc_hid_dim * (2 if bidirectional_encoder else 1)

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Input to LSTM: [embedding; context_vector]
        self.rnn = nn.LSTM(
            emb_dim + self.enc_total_dim,
            dec_hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

        # Output projection layer: takes [dec_hidden; context_vector; embedding] -> vocabulary logits
        self.fc_out = nn.Linear(dec_hid_dim + self.enc_total_dim + emb_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        enc_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Step-by-step single token decoding.
        Args:
            input_token: [batch_size] (current token input)
            hidden_state: (hidden [n_layers, batch_size, dec_hid_dim], cell [n_layers, batch_size, dec_hid_dim])
            enc_outputs: [batch_size, src_len, enc_total_dim]
            mask: [batch_size, src_len]
        Returns:
            prediction: [batch_size, output_dim]
            hidden_state: (new_hidden, new_cell)
            attention_weights: [batch_size, src_len]
        """
        hidden, cell = hidden_state

        # input_token: [batch_size] -> [batch_size, 1]
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        # Compute attention weights using top-layer decoder hidden state
        # hidden[-1]: [batch_size, dec_hid_dim]
        a = self.attention(hidden[-1], enc_outputs, mask)  # [batch_size, src_len]
        a_expanded = a.unsqueeze(1)  # [batch_size, 1, src_len]

        # Compute weighted context vector: [batch_size, 1, src_len] x [batch_size, src_len, enc_total_dim]
        # -> [batch_size, 1, enc_total_dim]
        context = torch.bmm(a_expanded, enc_outputs)

        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + enc_total_dim]

        # Forward through LSTM
        rnn_output, (new_hidden, new_cell) = self.rnn(rnn_input, (hidden, cell))
        # rnn_output: [batch_size, 1, dec_hid_dim]

        # Squeeze sequence dimension
        embedded = embedded.squeeze(1)      # [batch_size, emb_dim]
        rnn_output = rnn_output.squeeze(1)  # [batch_size, dec_hid_dim]
        context = context.squeeze(1)        # [batch_size, enc_total_dim]

        # Linear projection to vocabulary size
        prediction = self.fc_out(torch.cat((rnn_output, context, embedded), dim=1))  # [batch_size, output_dim]

        return prediction, (new_hidden, new_cell), a
