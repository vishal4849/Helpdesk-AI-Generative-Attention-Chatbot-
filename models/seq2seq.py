import random
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import SOS_IDX, EOS_IDX, PAD_IDX


class Seq2SeqAttention(nn.Module):
    """
    End-to-End Sequence-to-Sequence model with Bahdanau/Luong Attention Mechanism.
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_idx: int = PAD_IDX,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device if device is not None else torch.device("cpu")

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create binary mask where 1 = valid token and 0 = PAD."""
        mask = (src != self.pad_idx).to(self.device)
        return mask

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass during training with Teacher Forcing.
        Args:
            src: [batch_size, src_len]
            trg: [batch_size, trg_len]
            src_lengths: [batch_size]
            teacher_forcing_ratio: Probability of using ground truth token vs model prediction
        Returns:
            outputs: [batch_size, trg_len, output_dim]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encoder forward pass
        enc_outputs, hidden_state = self.encoder(src, src_lengths)
        mask = self.create_mask(src)

        # First input token to decoder is <SOS>
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden_state, _ = self.decoder(input_token, hidden_state, enc_outputs, mask)
            outputs[:, t, :] = output

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    def generate_response(
        self,
        src_indices: List[int],
        max_len: int = 30,
        temperature: float = 1.0,
        deterministic: bool = True
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Inference loop to generate output tokens step-by-step.
        Args:
            src_indices: List of integer token indices for source sentence.
            max_len: Maximum response length.
            temperature: Sampling temperature (>0). Lower = more deterministic/confident.
            deterministic: If True, uses greedy argmax decoding; otherwise multinomial sampling.
        Returns:
            predicted_indices: List of token IDs
            attention_matrix: [trg_len, src_len] list of attention distributions
        """
        self.eval()
        with torch.no_grad():
            src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)

            enc_outputs, hidden_state = self.encoder(src_tensor, src_lengths)
            mask = self.create_mask(src_tensor)

            # Start token <SOS>
            input_token = torch.tensor([SOS_IDX], dtype=torch.long).to(self.device)

            predicted_indices = []
            attention_matrix = []

            for _ in range(max_len):
                output, hidden_state, attention_weights = self.decoder(
                    input_token, hidden_state, enc_outputs, mask
                )
                attention_matrix.append(attention_weights.squeeze(0).cpu().tolist())

                if deterministic or temperature <= 0.01:
                    top_token = output.argmax(1).item()
                else:
                    scaled_logits = output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    top_token = torch.multinomial(probs, num_samples=1).item()

                if top_token == EOS_IDX:
                    break

                predicted_indices.append(top_token)
                input_token = torch.tensor([top_token], dtype=torch.long).to(self.device)

            return predicted_indices, attention_matrix
