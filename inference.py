import os
import torch
from typing import Tuple, List, Optional
from utils.text_preprocessing import normalize_text
from utils.vocabulary import Vocabulary, PAD_IDX
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2SeqAttention


class ChatbotInference:
    """
    Inference Engine for Seq2Seq Attention Chatbot.
    """
    def __init__(
        self,
        model_path: str = "saved_models/chatbot_model.pt",
        vocab_path: str = "saved_models/vocab.json",
        device_str: str = "auto"
    ):
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # 1. Load Vocabulary
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please train the model first.")
        self.vocab = Vocabulary.load_from_file(vocab_path)

        # 2. Load Model Checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")
        checkpoint = torch.load(model_path, map_location=self.device)
        cfg = checkpoint["config"]

        # 3. Instantiate model architecture
        encoder = Encoder(
            input_dim=cfg["input_dim"],
            emb_dim=cfg["emb_dim"],
            enc_hid_dim=cfg["enc_hid_dim"],
            dec_hid_dim=cfg["dec_hid_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg.get("bidirectional", True)
        )

        attention = Attention(
            enc_hid_dim=cfg["enc_hid_dim"],
            dec_hid_dim=cfg["dec_hid_dim"],
            attention_type=cfg.get("attention_type", "bahdanau"),
            bidirectional_encoder=cfg.get("bidirectional", True)
        )

        decoder = Decoder(
            output_dim=cfg["output_dim"],
            emb_dim=cfg["emb_dim"],
            enc_hid_dim=cfg["enc_hid_dim"],
            dec_hid_dim=cfg["dec_hid_dim"],
            attention=attention,
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            bidirectional_encoder=cfg.get("bidirectional", True)
        )

        self.model = Seq2SeqAttention(encoder, decoder, pad_idx=PAD_IDX, device=self.device).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def respond(
        self,
        user_input: str,
        max_len: int = 25,
        temperature: float = 0.7,
        deterministic: bool = True
    ) -> Tuple[str, List[str], List[str], List[List[float]]]:
        """
        Generate response for user input with attention matrix for visualization.
        
        Returns:
            response_text: string response
            input_tokens: list of cleaned input words + <EOS>
            output_tokens: list of generated words
            attention_matrix: 2D list [output_len, input_len]
        """
        cleaned_input = normalize_text(user_input)
        if not cleaned_input:
            return "Could you please type a message?", [], [], []

        # Convert to token indices with <EOS>
        src_indices = self.vocab.sentence_to_indices(cleaned_input, add_sos=False, add_eos=True)
        input_tokens = cleaned_input.split() + ["<EOS>"]

        predicted_indices, attention_matrix = self.model.generate_response(
            src_indices=src_indices,
            max_len=max_len,
            temperature=temperature,
            deterministic=deterministic
        )

        output_tokens = [self.vocab.index2word.get(idx, "<UNK>") for idx in predicted_indices]
        response_text = self.vocab.indices_to_sentence(predicted_indices, remove_special=True)

        if not response_text:
            response_text = "I am not sure I understand. Could you please rephrase?"

        # Capitalize first letter and format
        response_text = response_text[0].upper() + response_text[1:] if len(response_text) > 0 else response_text

        return response_text, input_tokens, output_tokens, attention_matrix


def interactive_cli():
    """Interactive command-line chat session."""
    print("=" * 60)
    print(" HelpDesk AI - Seq2Seq Attention Chatbot CLI")
    print(" Type 'exit' or 'quit' to terminate the session.")
    print("=" * 60)

    try:
        bot = ChatbotInference()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        try:
            user_msg = input("\nYou: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit", "q"]:
                print("HelpDesk AI: Goodbye! Have a great day!")
                break

            reply, in_tokens, out_tokens, _ = bot.respond(user_msg, deterministic=True)
            print(f"HelpDesk AI: {reply}")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break


if __name__ == "__main__":
    interactive_cli()
