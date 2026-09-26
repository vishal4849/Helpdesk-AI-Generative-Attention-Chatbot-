import os
import time
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_loader import build_dialogue_pipeline
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2SeqAttention
from utils.vocabulary import PAD_IDX
from utils.visualizer import plot_loss_perplexity


def train_epoch(
    model: Seq2SeqAttention,
    dataloader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip: float,
    teacher_forcing_ratio: float,
    device: torch.device
) -> float:
    model.train()
    epoch_loss = 0

    for src, trg, src_lengths, _ in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()

        # output: [batch_size, trg_len, output_dim]
        output = model(src, trg, src_lengths, teacher_forcing_ratio=teacher_forcing_ratio)

        output_dim = output.shape[-1]
        # Ignore first token (<SOS>) when calculating loss
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(
    json_path: str = "data/conversations.json",
    save_dir: str = "saved_models",
    epochs: int = 50,
    batch_size: int = 16,
    emb_dim: int = 128,
    enc_hid_dim: int = 256,
    dec_hid_dim: int = 256,
    n_layers: int = 1,
    dropout: float = 0.2,
    lr: float = 0.001,
    teacher_forcing_ratio: float = 0.6,
    attention_type: str = "bahdanau",
    clip: float = 1.0,
    device_str: str = "auto"
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[*] Training Device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Prepare data and vocabulary
    print("[*] Loading dataset and constructing vocabulary...")
    dataloader, vocab, pairs = build_dialogue_pipeline(
        json_path=json_path,
        batch_size=batch_size,
        max_len=25
    )
    print(f"[*] Loaded {len(pairs)} dialogue pairs. Vocabulary size: {len(vocab)} tokens.")

    vocab_file = os.path.join(save_dir, "vocab.json")
    vocab.save_to_file(vocab_file)
    print(f"[*] Saved vocabulary to {vocab_file}")

    # 2. Build model architecture
    print(f"[*] Initializing Seq2Seq Architecture with {attention_type.capitalize()} Attention...")
    encoder = Encoder(
        input_dim=len(vocab),
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional=True
    )

    attention = Attention(
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        attention_type=attention_type,
        bidirectional_encoder=True
    )

    decoder = Decoder(
        output_dim=len(vocab),
        emb_dim=emb_dim,
        enc_hid_dim=enc_hid_dim,
        dec_hid_dim=dec_hid_dim,
        attention=attention,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional_encoder=True
    )

    model = Seq2SeqAttention(encoder, decoder, pad_idx=PAD_IDX, device=device).to(device)

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {
        "train_loss": [],
        "train_ppl": []
    }

    print(f"[*] Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            clip=clip,
            teacher_forcing_ratio=teacher_forcing_ratio,
            device=device
        )
        ppl = math.exp(min(loss, 100))
        history["train_loss"].append(loss)
        history["train_ppl"].append(ppl)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch: {epoch:02d}/{epochs:02d} | Loss: {loss:.4f} | Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")

    # 4. Save checkpoint & metrics
    model_save_path = os.path.join(save_dir, "chatbot_model.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": len(vocab),
            "output_dim": len(vocab),
            "emb_dim": emb_dim,
            "enc_hid_dim": enc_hid_dim,
            "dec_hid_dim": dec_hid_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "attention_type": attention_type,
            "bidirectional": True
        }
    }
    torch.save(checkpoint, model_save_path)
    print(f"[*] Model weights saved to {model_save_path}")

    # Save training history JSON
    history_file = os.path.join(save_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot and save curves
    plot_file = os.path.join(save_dir, "training_curves.png")
    plot_loss_perplexity(history, save_path=plot_file)
    print(f"[*] Saved loss & perplexity plots to {plot_file}")

    return model, vocab, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Seq Attention Chatbot")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--enc_hid_dim", type=int, default=256, help="Encoder hidden dimension")
    parser.add_argument("--dec_hid_dim", type=int, default=256, help="Decoder hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--teacher_forcing", type=float, default=0.6, help="Teacher forcing ratio")
    parser.add_argument("--attention", type=str, default="bahdanau", choices=["bahdanau", "luong"], help="Attention type")
    parser.add_argument("--data_path", type=str, default="data/conversations.json", help="Path to conversations dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save model")

    args = parser.parse_args()

    train_model(
        json_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        lr=args.lr,
        teacher_forcing_ratio=args.teacher_forcing,
        attention_type=args.attention
    )
