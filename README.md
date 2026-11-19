# 🤖 HelpDesk AI: Generative Attention Chatbot & Recommendation System

An end-to-end deep learning conversational AI platform implemented in **PyTorch**, utilizing a **Bidirectional LSTM Encoder**, **Bahdanau Additive Attention Mechanism**, **Teacher Forcing**, and an **Interactive Modern Web Application** with real-time word-level attention heatmaps, tokenizer sandbox, and architecture simulation.

---

## 📋 Project Highlights & Deliverables

| Deliverable | Description | Location / Status |
| :--- | :--- | :--- |
| **1. The Model Notebook** | Full Jupyter Notebook with data preprocessing, dynamic batch padding, bidirectional encoder, Bahdanau attention, custom training loop, and convergence plots. | [`chatbot_seq2seq_attention.ipynb`](./chatbot_seq2seq_attention.ipynb) |
| **2. Trained Model Checkpoint** | Pre-trained PyTorch weights, serialized vocabulary mapping, and convergence history. | [`saved_models/chatbot_model.pt`](./saved_models/chatbot_model.pt), [`saved_models/vocab.json`](./saved_models/vocab.json) |
| **3. Interactive Web Platform** | Modern AI SaaS web application featuring streaming chat, live attention alignment heatmaps, tokenizer lab, and tensor inspector. | [`app.py`](./app.py) |
| **4. Recommendation Engine** | Intelligent recommendation layer for plans, troubleshooting diagnostics, and natural detokenization. | [`utils/recommendation_agent.py`](./utils/recommendation_agent.py) |
| **5. Modular Architecture** | Clean, production-ready modules for dataset collation, models, and inference CLI. | [`models/`](./models/), [`data/`](./data/), [`utils/`](./utils/), [`inference.py`](./inference.py) |

---

## 🏛️ Deep Learning Architecture

```
User Query (e.g. "How do I reset password?")
         │
         ▼
[ Word Embedding (128-d) ]
         │
         ▼
[ Bidirectional LSTM Encoder (256-d) ] ───► Encoder Hidden States {h_1, h_2, ... h_T} (512-d)
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

1. **Bahdanau Additive Attention Energy**:
   $$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

2. **Softmax Attention Weights**:
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

3. **Dynamic Context Vector**:
   $$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

4. **Decoder State Update**:
   $$s_i = \text{LSTM}([E(y_{i-1}); c_i], s_{i-1})$$

5. **Output Token Probability**:
   $$P(y_i \mid y_{<i}, \mathbf{x}) = \text{softmax}(W_o [s_i; c_i; E(y_{i-1})])$$

---

## 🌟 Interactive Web Platform Features

1. **💬 Live AI Chatbot Playground**:
   - Token-by-token typewriter streaming output.
   - Dual Mode: **Smart Production Agent** (hybrid recommendation & natural formatting) vs **Pure PyTorch Model** (raw neural output).
   - Dynamic prompt starters and follow-up chips.
   - In-message expandable word-level attention inspection.

2. **🔍 Attention Alignment Studio**:
   - High-resolution Bahdanau attention weight heatmap matrix.
   - Step-by-step decoder alignment breakdown with numerical confidence percentages.

3. **🔬 Preprocessing & Tokenizer Sandbox**:
   - Interactive testing of Unicode normalization, contraction expansion (`can't` $\rightarrow$ `cannot`), and regex cleaning.
   - Token-to-ID lookup highlighting special tokens (`<SOS>`, `<EOS>`, `<UNK>`).
   - Dynamic batch padding tensor simulation.

4. **🧠 Neural Architecture & Tensor Inspector**:
   - Interactive sliders for Batch Size ($B$) and Sequence Length ($T_x$) calculating exact tensor dimensions across every layer.
   - Trainable parameter breakdown (~2.6M params).

5. **📈 Training Convergence & Benchmarks**:
   - Loss & Perplexity reduction curves across 60 epochs ($373.1 \rightarrow 1.033$).
   - Comparative ablation benchmark table (Bahdanau Attention vs Luong Attention vs Vanilla Seq2Seq).

6. **📋 Hackathon Deliverables Scorecard**:
   - Formal verification matrix mapping each specification requirement to implementation status.

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/vishal4849/Helpdesk-AI-Generative-Attention-Chatbot-.git
cd Helpdesk-AI-Generative-Attention-Chatbot-
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Interactive Command-Line Chat (CLI)
Chat directly in your terminal:
```bash
python inference.py
```

### 4. Retrain the Model (Optional)
The pre-trained weights are already included in `saved_models/chatbot_model.pt`. To retrain:
```bash
python train.py --epochs 60 --batch_size 16 --emb_dim 128 --enc_hid_dim 256 --dec_hid_dim 256 --teacher_forcing 0.6 --attention bahdanau
```

---

## 📊 Performance & Convergence Metrics

- **Final Cross-Entropy Loss**: `0.033` (🔻 -99.4% from Epoch 1)
- **Final Perplexity (PPL)**: `1.033` (🔻 -99.7% from 373.1, meeting target `< 1.05`)
- **Vocabulary Size**: `382 unique tokens`
- **Special Tokens**: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`
- **Embedding Dimension**: `128`
- **Encoder Hidden Units**: `256` (Bidirectional $\rightarrow$ `512`)
- **Decoder Hidden Units**: `256`
- **Total Parameters**: `~2,599,000`

---

## 📁 Repository Structure

```
├── app.py                                # Modern Streamlit Web Application
├── inference.py                          # Command-Line Chatbot Interface
├── train.py                              # PyTorch Training Loop with Teacher Forcing
├── requirements.txt                      # Project Dependencies
├── chatbot_seq2seq_attention.ipynb       # End-to-end Jupyter Notebook
├── data/
│   ├── conversations.json                # Conversational Dataset Pairs
│   └── dataset_loader.py                 # Dynamic Padding Dataset & DataLoader
├── models/
│   ├── encoder.py                        # Bidirectional LSTM Encoder
│   ├── attention.py                      # Bahdanau & Luong Attention Layer
│   ├── decoder.py                        # Attention-conditioned LSTM Decoder
│   └── seq2seq.py                        # Seq2Seq Model Wrapper & Inference Loop
├── utils/
│   ├── text_preprocessing.py             # Normalization, Contractions, Cleaning
│   ├── vocabulary.py                     # Vocabulary Class & Serialization
│   ├── visualizer.py                     # Heatmap & Curve Plotting
│   └── recommendation_agent.py           # Conversational Recommendation Layer
└── saved_models/
    ├── chatbot_model.pt                  # Serialized Model Weights
    ├── vocab.json                        # Serialized Vocabulary Mapping
    ├── training_history.json             # Epoch Loss & Perplexity History
    └── training_curves.png               # High-res Loss & Perplexity Plots
```

---

## 📄 License
MIT License
