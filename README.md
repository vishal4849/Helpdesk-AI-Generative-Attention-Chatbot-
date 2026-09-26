# 🤖 HelpDesk AI: Chatbot using Sequence-to-Sequence (Seq2Seq) with Attention Mechanism

An end-to-end deep learning generative customer support chatbot implemented in **PyTorch**, utilizing a **Bidirectional LSTM Encoder**, **Bahdanau & Luong Attention Mechanisms**, **Teacher Forcing**, and a **Modern WhatsApp-styled Streamlit Web Application** with live attention alignment heatmap visualizer.

---

## 📋 Project Summary & Deliverables

As outlined in the Persevex Project specification (*chatbot-seq2seq-attention-mechanism*):

| Deliverable | Description | Location / Status |
| :--- | :--- | :--- |
| **1. The Model Notebook** | Full Jupyter Notebook containing preprocessing, dynamic padding, bidirectional encoder, Bahdanau attention layer, decoder, custom training loop, and convergence plots. | [`chatbot_seq2seq_attention.ipynb`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/chatbot_seq2seq_attention.ipynb) |
| **2. The Chatbot Model** | Trained PyTorch weights, serialized vocabulary mapping, and epoch history metrics. | [`saved_models/chatbot_model.pt`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/saved_models/chatbot_model.pt), [`saved_models/vocab.json`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/saved_models/vocab.json), [`saved_models/training_history.json`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/saved_models/training_history.json) |
| **3. Demo Web Interface** | Modern WhatsApp-style Streamlit web application with live interactive chat, real-time word-level attention heatmaps, and architecture inspection. | [`app.py`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/app.py) |
| **4. Core Modules** | Modular, production-ready Python packages for dataset loading, preprocessing, model components, and inference CLI. | [`models/`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/models/), [`data/`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/data/), [`utils/`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/utils/), [`inference.py`](file:///c:/Users/sarth/OneDrive/Desktop/Chatbot%20Recommendation%20Persevex/inference.py) |

---

## 🏛️ Deep Learning Architecture

```
User Query (e.g. "How do I reset password?")
         │
         ▼
[ Word Embedding (128-d) ]
         │
         ▼
[ Bidirectional LSTM Encoder (256-d) ] ───► Encoder Hidden States {h_1, h_2, ... h_T}
                                                     │
                                                     ▼
                                          [ Bahdanau Attention ] ◄── Previous Decoder State (s_{t-1})
                                                     │
                                            Attention Weights α_t
                                                     │
                                            Context Vector c_t
                                                     │
                                                     ▼
                                          [ Decoder LSTM (256-d) ] ◄── Previous Output Token y_{t-1}
                                                     │
                                                     ▼
                                          [ Linear Classifier ]
                                                     │
                                                     ▼
                                   Output Token Prediction y_t (e.g. "Click", "on", "forgot", ...)
```

### Mathematical Foundations

1. **Bahdanau Additive Attention Score**:
   $$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$
   $$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

2. **Decoder State Update**:
   $$s_i = \text{LSTM}([E(y_{i-1}); c_i], s_{i-1})$$

3. **Output Token Probability**:
   $$P(y_i \mid y_{<i}, \mathbf{x}) = \text{softmax}(W_o [s_i; c_i; E(y_{i-1})])$$

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure dependencies are installed:
```powershell
pip install -r requirements.txt
```

### 2. Train the Model (Optional - Already Pre-trained)
To retrain or adjust hyperparameters:
```powershell
python train.py --epochs 60 --batch_size 16 --emb_dim 128 --enc_hid_dim 256 --dec_hid_dim 256 --teacher_forcing 0.6 --attention bahdanau
```
Training generates:
- Model checkpoint: `saved_models/chatbot_model.pt`
- Vocabulary JSON: `saved_models/vocab.json`
- Convergence Plots: `saved_models/training_curves.png`

### 3. Launch the Modern WhatsApp-Style Web App
Run the Streamlit application:
```powershell
streamlit run app.py
```

### 4. Interactive Command-Line Chat
Chat directly in your terminal:
```powershell
python inference.py
```

---

## 📊 Training Results & Performance

- **Final Cross-Entropy Loss**: `~0.033`
- **Perplexity (PPL)**: `1.03`
- **Vocabulary Size**: `382 unique tokens`
- **Supported Special Tokens**: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`
