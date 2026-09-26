import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from utils.text_preprocessing import normalize_text, filter_pair
from utils.vocabulary import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


class DialogueDataset(Dataset):
    """
    PyTorch Dataset for conversational pairs (Question -> Answer).
    """
    def __init__(self, pairs: List[Tuple[str, str]], vocab: Vocabulary, max_len: int = 20):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

        # Pre-convert to tensors
        self.processed_pairs = []
        for q, a in self.pairs:
            src_indices = self.vocab.sentence_to_indices(q, add_sos=False, add_eos=True)[:self.max_len]
            trg_indices = self.vocab.sentence_to_indices(a, add_sos=True, add_eos=True)[:self.max_len]
            self.processed_pairs.append((
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(trg_indices, dtype=torch.long)
            ))

    def __len__(self) -> int:
        return len(self.processed_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processed_pairs[idx]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to dynamically pad sequences in a batch.
    Returns:
        src_padded: [batch_size, max_src_len]
        trg_padded: [batch_size, max_trg_len]
        src_lengths: [batch_size]
        trg_lengths: [batch_size]
    """
    src_seqs, trg_seqs = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    trg_lengths = torch.tensor([len(t) for t in trg_seqs], dtype=torch.long)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True, padding_value=PAD_IDX)

    return src_padded, trg_padded, src_lengths, trg_lengths


def load_pairs_from_json(json_path: str, max_len: int = 25) -> List[Tuple[str, str]]:
    """Load and normalize dialogue pairs from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        q = normalize_text(item["question"])
        a = normalize_text(item["answer"])
        if filter_pair(q, a, min_len=1, max_len=max_len):
            pairs.append((q, a))
    return pairs


def build_dialogue_pipeline(
    json_path: str,
    batch_size: int = 16,
    max_len: int = 25,
    min_word_count: int = 1,
    existing_vocab: Optional[Vocabulary] = None
) -> Tuple[DataLoader, Vocabulary, List[Tuple[str, str]]]:
    """
    Full pipeline to load raw text, build vocabulary, create dataset and dataloader.
    """
    pairs = load_pairs_from_json(json_path, max_len=max_len)

    if existing_vocab is not None:
        vocab = existing_vocab
    else:
        vocab = Vocabulary(name="helpdesk_vocab")
        for q, a in pairs:
            vocab.add_sentence(q)
            vocab.add_sentence(a)
        if min_word_count > 1:
            vocab.trim(min_word_count)

    dataset = DialogueDataset(pairs, vocab, max_len=max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    return dataloader, vocab, pairs
