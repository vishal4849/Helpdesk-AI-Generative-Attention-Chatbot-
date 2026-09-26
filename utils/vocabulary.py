import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Special Tokens
PAD_TOKEN = "<PAD>"  # Index 0
SOS_TOKEN = "<SOS>"  # Index 1
EOS_TOKEN = "<EOS>"  # Index 2
UNK_TOKEN = "<UNK>"  # Index 3

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """
    Vocabulary class to map tokens to indices and vice-versa,
    supporting frequency-based word filtering, serialization and deserialization.
    """
    def __init__(self, name: str = "default_vocab"):
        self.name = name
        self.word2index: Dict[str, int] = {}
        self.word2count: Dict[str, int] = Counter()
        self.index2word: Dict[int, str] = {}
        self.num_words: int = 0
        self._reset()

    def _reset(self):
        """Initialize vocabulary with special tokens."""
        self.word2index = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX
        }
        self.index2word = {idx: word for word, idx in self.word2index.items()}
        self.word2count = Counter()
        self.num_words = 4

    def add_sentence(self, sentence: str):
        """Tokenize sentence and add all words into vocabulary frequency tracker."""
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word: str):
        """Add individual word to vocabulary."""
        self.word2count[word] += 1
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    def trim(self, min_count: int = 1):
        """
        Trim words below a certain frequency count to control vocabulary size.
        Re-indexes the vocabulary.
        """
        if min_count <= 1:
            return

        keep_words = [word for word, count in self.word2count.items() if count >= min_count]
        
        # Reset and recreate
        self._reset()
        for word in keep_words:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    def sentence_to_indices(self, sentence: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a sentence into a list of token indices."""
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for word in sentence.split():
            indices.append(self.word2index.get(word, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def indices_to_sentence(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert a list of indices back to text."""
        words = []
        for idx in indices:
            if idx == EOS_IDX and remove_special:
                break
            if remove_special and idx in (PAD_IDX, SOS_IDX):
                continue
            words.append(self.index2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def save_to_file(self, filepath: str):
        """Save vocabulary mapping and word counts to JSON."""
        data = {
            "name": self.name,
            "word2index": self.word2index,
            "index2word": {str(k): v for k, v in self.index2word.items()},
            "word2count": dict(self.word2count),
            "num_words": self.num_words
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary mapping from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(name=data.get("name", "loaded_vocab"))
        vocab.word2index = data["word2index"]
        vocab.index2word = {int(k): v for k, v in data["index2word"].items()}
        vocab.word2count = Counter(data.get("word2count", {}))
        vocab.num_words = data["num_words"]
        return vocab

    def __len__(self):
        return self.num_words
