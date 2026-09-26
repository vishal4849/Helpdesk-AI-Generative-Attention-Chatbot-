import torch
import torch.nn as nn
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Encoder module for Seq2Seq architecture.
    Uses an Embedding layer followed by a multi-layer Bidirectional/Unidirectional LSTM.
    """
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        n_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.rnn = nn.LSTM(
            emb_dim,
            enc_hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Linear projections from encoder hidden states to decoder hidden states
        total_enc_dim = enc_hid_dim * self.num_directions
        self.fc_hidden = nn.Linear(total_enc_dim, dec_hid_dim)
        self.fc_cell = nn.Linear(total_enc_dim, dec_hid_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [batch_size, src_len]
            src_lengths: [batch_size] optional lengths for packing
        Returns:
            outputs: [batch_size, src_len, enc_hid_dim * num_directions]
            (hidden, cell): each [n_layers, batch_size, dec_hid_dim]
        """
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]

        if src_lengths is not None:
            # Pack sequence for computational efficiency and handling padding
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                src_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs,
                batch_first=True
            )
        else:
            outputs, (hidden, cell) = self.rnn(embedded)

        # outputs: [batch_size, src_len, enc_hid_dim * num_directions]
        # hidden, cell: [n_layers * num_directions, batch_size, enc_hid_dim]
        
        # Bridge encoder state to decoder dimension
        if self.bidirectional:
            # Concatenate top-layer forward and backward states
            # hidden[-2, :, :] is forward, hidden[-1, :, :] is backward
            hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, enc_hid_dim * 2]
            cell_cat = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

            dec_init_hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)  # [1, batch_size, dec_hid_dim]
            dec_init_cell = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)        # [1, batch_size, dec_hid_dim]
        else:
            dec_init_hidden = torch.tanh(self.fc_hidden(hidden[-1, :, :])).unsqueeze(0)
            dec_init_cell = torch.tanh(self.fc_cell(cell[-1, :, :])).unsqueeze(0)

        # Expand to n_layers if decoder has > 1 layers
        if self.n_layers > 1:
            dec_init_hidden = dec_init_hidden.repeat(self.n_layers, 1, 1)
            dec_init_cell = dec_init_cell.repeat(self.n_layers, 1, 1)

        return outputs, (dec_init_hidden, dec_init_cell)
