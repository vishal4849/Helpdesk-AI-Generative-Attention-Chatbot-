import os
import json
import time
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from inference import ChatbotInference
from utils.text_preprocessing import normalize_text

# Page configuration
st.set_page_config(
    page_title="HelpDesk AI | Seq2Seq Attention Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Messaging & Glassmorphism UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #075e54 0%, #128c7e 50%, #25d366 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(18, 140, 126, 0.2);
    }
    
    .chat-container {
        background: #efeae2;
        background-image: radial-gradient(#d1d7db 1px, transparent 1px);
        background-size: 20px 20px;
        border-radius: 16px;
        padding: 20px;
        height: 520px;
        overflow-y: auto;
        border: 1px solid #d1d7db;
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 15px;
    }
    
    .user-bubble {
        background-color: #d9fdd3;
        color: #111b21;
        padding: 10px 14px;
        border-radius: 12px 0px 12px 12px;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        font-size: 14.5px;
        line-height: 1.4;
        position: relative;
    }
    
    .bot-bubble {
        background-color: #ffffff;
        color: #111b21;
        padding: 10px 14px;
        border-radius: 0px 12px 12px 12px;
        max-width: 75%;
        margin-right: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        font-size: 14.5px;
        line-height: 1.4;
        position: relative;
    }
    
    .meta-time {
        font-size: 10px;
        color: #667781;
        text-align: right;
        margin-top: 4px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    
    .chip-btn {
        background-color: #f0f2f5;
        border: 1px solid #d1d7db;
        border-radius: 16px;
        padding: 4px 10px;
        font-size: 12px;
        cursor: pointer;
        display: inline-block;
        margin: 3px;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot_engine():
    """Cache and load inference engine."""
    return ChatbotInference(
        model_path="saved_models/chatbot_model.pt",
        vocab_path="saved_models/vocab.json"
    )


try:
    bot = load_chatbot_engine()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model or vocabulary not found. Please train the model first by running `python train.py`. Error: {e}")

# Initialize Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "sender": "bot",
            "text": "Hello! I am HelpDesk AI, your intelligent customer support assistant. How can I help you today?",
            "time": datetime.datetime.now().strftime("%I:%M %p")
        }
    ]

if "last_attention" not in st.session_state:
    st.session_state.last_attention = None
    st.session_state.last_in_tokens = []
    st.session_state.last_out_tokens = []

# Sidebar Controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=64)
    st.title("HelpDesk AI Settings")
    st.markdown("Sequence-to-Sequence (Seq2Seq) with **Bahdanau Attention**")
    st.divider()

    st.subheader("⚙️ Decoding Parameters")
    decoding_mode = st.radio("Decoding Mode", ["Greedy (Argmax)", "Sampling (Temperature)"])
    deterministic = (decoding_mode == "Greedy (Argmax)")
    
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1, disabled=deterministic)
    max_len = st.slider("Max Response Length", min_value=5, max_value=40, value=25, step=1)
    
    st.divider()
    st.subheader("💡 Quick Query Chips")
    quick_queries = [
        "Hi",
        "Who are you?",
        "How do I reset my password?",
        "How much does it cost?",
        "The app crashed",
        "What is Seq2Seq?",
        "How does attention mechanism work?",
        "Can I talk to a human?",
        "Thank you"
    ]
    
    selected_chip = None
    for q in quick_queries:
        if st.button(f"💬 {q}", key=f"chip_{q}", use_container_width=True):
            selected_chip = q

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "sender": "bot",
                "text": "Chat reset. How can I help you today?",
                "time": datetime.datetime.now().strftime("%I:%M %p")
            }
        ]
        st.session_state.last_attention = None
        st.rerun()

# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h2 style="margin:0; font-size: 1.8rem; font-weight: 700;">💬 HelpDesk AI — Generative Chatbot</h2>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                Powered by Deep Learning Seq2Seq (Encoder-Decoder LSTM) + Bahdanau Attention Mechanism
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(255,255,255,0.25); padding: 5px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                🟢 Online & Ready
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_chat, tab_attention, tab_architecture, tab_dataset = st.tabs([
    "💬 WhatsApp-Style Chat",
    "🔍 Attention Heatmap",
    "🧠 Architecture & Performance",
    "📚 Dataset & Vocabulary"
])

# ----------------- TAB 1: Live Chat Interface -----------------
with tab_chat:
    # Render Chat History
    chat_html = ['<div class="chat-container">']
    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            chat_html.append(f"""
            <div class="user-bubble">
                <div>{msg["text"]}</div>
                <div class="meta-time">{msg["time"]} &nbsp; ✓✓</div>
            </div>
            """)
        else:
            chat_html.append(f"""
            <div class="bot-bubble">
                <strong style="color: #075e54; font-size: 11px;">HelpDesk AI</strong><br/>
                <div>{msg["text"]}</div>
                <div class="meta-time">{msg["time"]}</div>
            </div>
            """)
    chat_html.append('</div>')
    st.markdown("".join(chat_html), unsafe_allow_html=True)

    # Input Form
    with st.form(key="chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_text_input = st.text_input(
                "Type your message here...",
                placeholder="Type 'Hi', 'How do I reset password', etc...",
                label_visibility="collapsed"
            )
        with col_btn:
            submit_btn = st.form_submit_with_button = st.form_submit_button("Send 🚀", use_container_width=True)

    # Handle Input or Quick Chip Click
    message_to_send = None
    if submit_btn and user_text_input.strip():
        message_to_send = user_text_input.strip()
    elif selected_chip:
        message_to_send = selected_chip

    if message_to_send and model_loaded:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        
        # Add user message
        st.session_state.messages.append({
            "sender": "user",
            "text": message_to_send,
            "time": current_time
        })

        # Generate bot response
        with st.spinner("HelpDesk AI is thinking..."):
            reply, in_tokens, out_tokens, attn_matrix = bot.respond(
                user_input=message_to_send,
                max_len=max_len,
                temperature=temperature,
                deterministic=deterministic
            )
            bot_time = datetime.datetime.now().strftime("%I:%M %p")
            
            # Save response & attention for heatmap tab
            st.session_state.messages.append({
                "sender": "bot",
                "text": reply,
                "time": bot_time
            })
            st.session_state.last_attention = attn_matrix
            st.session_state.last_in_tokens = in_tokens
            st.session_state.last_out_tokens = out_tokens

        st.rerun()

# ----------------- TAB 2: Attention Alignment Heatmap -----------------
with tab_attention:
    st.subheader("🔍 Bahdanau Attention Alignment Matrix")
    st.markdown("""
    The **Attention Mechanism** allows the decoder LSTM to 'look back' at specific words in the input query 
    at each step of word generation, assigning mathematical weights $\\alpha_{ij}$ via softmax:
    $$e_{ij} = v_a^T \\tanh(W_a s_{i-1} + U_a h_j), \\quad \\alpha_{ij} = \\frac{\\exp(e_{ij})}{\\sum_k \\exp(e_{ik})}$$
    """)

    if st.session_state.last_attention is not None and len(st.session_state.last_attention) > 0:
        attn_matrix = np.array(st.session_state.last_attention)
        in_tokens = st.session_state.last_in_tokens
        out_tokens = st.session_state.last_out_tokens

        fig, ax = plt.subplots(figsize=(max(7, len(in_tokens) * 0.9), max(5, len(out_tokens) * 0.55)))
        cax = ax.matshow(attn_matrix, cmap='YlGnBu')
        fig.colorbar(cax, shrink=0.8)

        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(out_tokens)))
        ax.set_xticklabels(in_tokens, rotation=45, ha='left', fontsize=11, fontweight='500')
        ax.set_yticklabels(out_tokens, fontsize=11, fontweight='500')

        ax.set_xlabel('Encoder Input Tokens', fontsize=12, labelpad=12, fontweight='bold', color='#075e54')
        ax.set_ylabel('Decoder Output Tokens', fontsize=12, labelpad=12, fontweight='bold', color='#075e54')
        plt.title('Real-Time Word-Level Attention Alignment', fontsize=13, pad=25, fontweight='bold')
        
        st.pyplot(fig)
        
        # Display Token Alignment Table
        with st.expander("📊 View Numerical Attention Probabilities Table"):
            df_attn = pd.DataFrame(attn_matrix, index=out_tokens, columns=in_tokens)
            st.dataframe(df_attn.style.background_gradient(cmap='Blues'), use_container_width=True)
    else:
        st.info("💡 Send a message in the chat tab to generate and inspect the live attention alignment heatmap!")

# ----------------- TAB 3: Model Architecture & Training Metrics -----------------
with tab_architecture:
    st.subheader("🧠 Deep Learning Architecture Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Encoder RNN", "Bidirectional LSTM", "256 Hidden Units")
    with col2:
        st.metric("Attention Layer", "Bahdanau (Additive)", "Context Vector")
    with col3:
        st.metric("Decoder RNN", "Unidirectional LSTM", "256 Hidden Units")
    with col4:
        if model_loaded:
            st.metric("Vocabulary Size", f"{len(bot.vocab)} Words", "Special Tokens: 4")

    st.markdown("---")
    
    st.subheader("📈 Training Convergence & Perplexity Curves")
    history_path = "saved_models/training_history.json"
    plot_path = "saved_models/training_curves.png"
    
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Cross-Entropy Loss and Perplexity Convergence across 60 Epochs", use_container_width=True)
    elif os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        df_hist = pd.DataFrame(history)
        st.line_chart(df_hist)

    st.markdown("---")
    st.subheader("📐 Mathematical Formulation")
    st.markdown("""
    | Component | Mathematical Operation | Purpose |
    | :--- | :--- | :--- |
    | **Encoder LSTM** | $h_t = \\text{BiLSTM}(E(x_t), h_{t-1})$ | Compresses input question into bidirectional hidden representations. |
    | **Attention Score** | $e_{ij} = v_a^T \\tanh(W_a s_{i-1} + U_a h_j)$ | Evaluates semantic relevance between decoder state $s_{i-1}$ and encoder step $h_j$. |
    | **Attention Weights** | $\\alpha_{ij} = \\text{softmax}(e_{ij})$ | Probability distribution over input tokens. |
    | **Context Vector** | $c_i = \\sum_j \\alpha_{ij} h_j$ | Dynamic weighted sum of encoder hidden states for step $i$. |
    | **Decoder LSTM** | $s_i = \\text{LSTM}([E(y_{i-1}); c_i], s_{i-1})$ | Generates next hidden state conditioning on previous word + attention context. |
    | **Projection Layer** | $P(y_i) = \\text{softmax}(W_o [s_i; c_i; E(y_{i-1})])$ | Computes probability distribution over the entire vocabulary. |
    """)

# ----------------- TAB 4: Dataset & Vocabulary -----------------
with tab_dataset:
    st.subheader("📚 Dataset & Preprocessing Pipeline")
    
    st.markdown("""
    **Preprocessing Pipeline Workflow:**
    1. **Text Normalization**: Unicode normalization to ASCII, lowercasing, whitespace collapsing.
    2. **Contraction Expansion**: E.g., `don't` $\\rightarrow$ `do not`, `what's` $\\rightarrow$ `what is`, `i'm` $\\rightarrow$ `i am`.
    3. **Symbol Cleaning**: Emojis, URLs, and non-alphanumeric symbols removed while preserving grammatical punctuation tokens.
    4. **Tokenization & Indexing**: Numerical conversion with `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`.
    5. **Dynamic Batch Padding**: `pad_sequence` collate function with length masks.
    """)

    conv_path = "data/conversations.json"
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            raw_convs = json.load(f)
        
        st.markdown(f"**Total Loaded Conversational Pairs:** `{len(raw_convs)}`")
        df_convs = pd.DataFrame(raw_convs)
        st.dataframe(df_convs, use_container_width=True, height=350)
