import re
import random
from typing import Tuple, List, Dict, Any, Optional

def natural_detokenize(text: str) -> str:
    """Format space-separated tokens into natural English typography."""
    if not text:
        return ""
    
    # Fix spaced punctuation
    text = re.sub(r'\s+([.,!?;:\'])', r'\1', text)
    # Fix contractions: "do n't" -> "don't", "can ' t" -> "can't"
    text = re.sub(r'(\w+)\s+\'\s*(\w+)', r'\1\'\2', text)
    text = re.sub(r'(\w+)\s+n\'t', r'\1n\'t', text)
    # Capitalize isolated 'i'
    text = re.sub(r'\b(i)\b', 'I', text)
    
    # Capitalize start of sentences
    sentences = re.split(r'([.!?]\s+)', text)
    formatted = []
    for s in sentences:
        if s and not re.match(r'^[.!?]\s+$', s):
            s = s.strip()
            if s:
                s = s[0].upper() + s[1:]
        formatted.append(s)
    res = "".join(formatted).strip()
    if res and len(res) > 0:
        res = res[0].upper() + res[1:]
    return res


def check_is_truncated(text: str) -> bool:
    """Check if generated text terminates abruptly in an incomplete preposition or article."""
    cleaned = text.lower().strip()
    # Strip terminal punctuation
    stripped = re.sub(r'[.!?]+$', '', cleaned).strip()
    dangling_words = (
        'of', 'the', 'to', 'in', 'for', 'with', 'and', 'a', 'an', 
        'or', 'at', 'by', 'from', 'on', 'about', 'as', 'into', 'like'
    )
    if any(stripped.endswith(' ' + w) or stripped == w for w in dangling_words):
        return True
    if len(stripped.split()) <= 2:
        return True
    if "<unk>" in cleaned:
        return True
    return False


class EnhancedRecommendationAgent:
    """
    Intelligent Conversational & Recommendation Layer for HelpDesk AI.
    Blends PyTorch Seq2Seq Bahdanau Attention representations with rich 
    domain-specific knowledge, natural detokenization, and multi-turn recommendation logic.
    """

    def __init__(self):
        self.knowledge_base = {
            "plans": {
                "keywords": ["plan", "pricing", "cost", "subscription", "price", "tier", "how much", "membership", "package"],
                "title": "💳 Subscription & Pricing Plans Recommendation",
                "content": """Here is our recommended breakdown of available plans:

• **Starter Plan ($10/mo)**:
  - Best for individual developers and small test projects.
  - Up to 1,000 monthly chatbot queries & standard response speed.

• **Professional Plan ($29/mo)** — ⭐ *Most Popular*:
  - Best for growing startups & active support teams.
  - 25,000 queries/month, real-time attention telemetry, and priority inference.

• **Enterprise Plan ($99/mo)**:
  - Best for organizations requiring custom security and fine-tuning.
  - Unlimited queries, custom model training, dedicated SLA support, and GPU hosting.

👉 *Annual billing includes a 20% discount on all plans!*""",
                "suggestions": ["How do I upgrade to Pro?", "What payment methods are supported?", "Is there a free trial?"]
            },
            "recommend_product": {
                "keywords": ["recommend", "recommendation", "which plan", "suggest", "what should i choose", "what do you suggest", "advice"],
                "title": "🎯 Personalized Recommendation",
                "content": """Based on typical use cases, here are our tailored recommendations:

1. **For Evaluation & Hackathon Presentations**:
   - The **Starter Tier** or Local Execution is ideal. It provides full access to the **Bidirectional LSTM + Bahdanau Attention** inference pipeline and live heatmaps.

2. **For High-Volume Customer Service Workloads**:
   - I recommend our **Professional Plan** ($29/mo). It features batch dynamic collation, optimized decoding, and sub-50ms inference latency.

3. **For Custom Enterprise Datasets**:
   - The **Enterprise Plan** allows uploading domain-specific conversational pairs in `data/conversations.json` and retraining the model weights with custom hyperparameters.""",
                "suggestions": ["Tell me about the Pro plan", "How to retrain the model?", "What is the accuracy?"]
            },
            "crash": {
                "keywords": ["crash", "crashing", "bug", "freeze", "error", "broken", "issue", "problem", "not working", "fails"],
                "title": "🛠️ Troubleshooting & Fix Recommendations",
                "content": """I am sorry to hear you are running into issues! Here are recommended steps to resolve it quickly:

1. **Verify Dependencies**:
   - Ensure `torch >= 2.0.0` and `streamlit >= 1.25.0` are properly installed.

2. **Clear Application Cache**:
   - In Streamlit: Press `C` in your browser to clear cache and restart.
   - Hard refresh your browser using `Ctrl + F5`.

3. **Verify Model Weights**:
   - Ensure `saved_models/chatbot_model.pt` exists and is intact (~10.4 MB).

4. **Inspect Console Logs**:
   - If an error trace appears in the terminal, check the attention tensor dimensions `[batch_size, seq_len]`.

If the issue persists, our technical team is available to assist!""",
                "suggestions": ["How do I view error logs?", "Can I talk to human support?", "Reset application"]
            },
            "password": {
                "keywords": ["password", "reset password", "forgot password", "unlock", "account locked"],
                "title": "🔐 Password Reset & Account Recovery",
                "content": """To securely reset or update your password:

1. Click on **Forgot Password?** on the login portal.
2. Enter your registered email address to receive a secure reset link.
3. Check your spam/junk folder if the link does not arrive within 2 minutes.
4. Passwords must be at least **8 characters** and include an uppercase letter, number, and symbol.

*Note: If your account was locked due to repeated attempts, it will auto-unlock in 15 minutes.*""",
                "suggestions": ["I didn't receive the email", "My account is still locked", "Contact security support"]
            },
            "attention_explain": {
                "keywords": ["attention", "bahdanau", "luong", "seq2seq", "bilstm", "encoder", "decoder", "how does it work"],
                "title": "🧠 Deep Learning Architecture & Attention Mechanism",
                "content": """**How the Bahdanau Attention Mechanism Works**:

Standard Seq2Seq architectures compress the entire input into a single fixed vector, which struggles on longer sentences. 

**Our Solution**:
1. **Bidirectional LSTM Encoder**: Encodes the source sentence forward and backward into states $h_j$ ($2 \\times 256 = 512$ dimensions).
2. **Bahdanau Additive Attention Layer**: Evaluates relevance between decoder state $s_{i-1}$ and all encoder states $h_j$:
   $$e_{ij} = v_a^T \\tanh(W_a s_{i-1} + U_a h_j)$$
3. **Softmax Weights**: Normalizes scores into probabilities $\\alpha_{ij} \\in [0, 1]$.
4. **Dynamic Context Vector**:
   $$c_i = \\sum_j \\alpha_{ij} h_j$$
   Provided directly to the Decoder LSTM at every decoding step $i$!""",
                "suggestions": ["Show Attention Heatmap", "What is the Perplexity score?", "Compare Bahdanau vs Luong"]
            },
            "about_bot": {
                "keywords": ["who are you", "what are you", "your name", "who made you", "what can you do"],
                "title": "🤖 About HelpDesk AI",
                "content": """Hello! I am **HelpDesk AI**, an enterprise customer support and recommendation chatbot.

**Core Capabilities**:
• **Generative Dialogue**: Powered by PyTorch Seq2Seq with Bidirectional LSTM.
• **Bahdanau Attention**: Dynamically aligns generated words with user query tokens.
• **Customer Care & Recommendations**: Handles account troubleshooting, password recovery, billing tiers, and technical support.
• **High Accuracy**: Trained to a low perplexity of **1.033** across 60 epochs.""",
                "suggestions": ["Recommend a plan for me", "How do I reset password?", "How does attention work?"]
            }
        }

    def process(
        self,
        user_query: str,
        raw_neural_reply: str,
        in_tokens: List[str],
        out_tokens: List[str],
        attn_matrix: List[List[float]],
        mode: str = "hybrid"
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process user query and return enhanced response, suggested chips, and metadata.
        """
        cleaned_query = user_query.lower().strip()
        detokenized_raw = natural_detokenize(raw_neural_reply)
        is_truncated = check_is_truncated(detokenized_raw)

        metadata = {
            "neural_source": "PyTorch BiLSTM + Bahdanau Attention",
            "is_enriched": False,
            "raw_output": raw_neural_reply,
            "token_count": len(out_tokens)
        }

        # If in pure neural mode, return detokenized raw model generation
        if mode == "pure_neural":
            suggestions = ["How do I reset password?", "How much does it cost?", "The app crashed"]
            return detokenized_raw, suggestions, metadata

        # Hybrid intelligent mode: inspect semantic intents
        matched_kb = None
        for key, entry in self.knowledge_base.items():
            for kw in entry["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', cleaned_query):
                    matched_kb = entry
                    break
            if matched_kb:
                break

        # If query asks for recommendation or specific guidance, or raw output was truncated:
        if matched_kb:
            metadata["is_enriched"] = True
            metadata["category"] = matched_kb["title"]
            
            # If the raw response is clean and not truncated, prepend it; otherwise use the curated content
            if not is_truncated and len(detokenized_raw.split()) >= 4:
                response = f"{detokenized_raw}\n\n{matched_kb['content']}"
            else:
                response = matched_kb["content"]
            
            return response, matched_kb["suggestions"], metadata

        # General greetings / small talk enhancement
        if re.match(r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b', cleaned_query):
            response = "Hello there! 👋 I am **HelpDesk AI**, your intelligent customer support and recommendation assistant. How can I help you today?"
            suggestions = ["Recommend a subscription plan", "How do I reset my password?", "How does attention work?"]
            return response, suggestions, metadata

        if re.search(r'\b(thank\s+you|thanks|thx)\b', cleaned_query):
            response = "You are very welcome! 😊 It is my pleasure to help. Feel free to ask if you have any more questions or need recommendations!"
            suggestions = ["Tell me about pricing", "Troubleshoot an app issue", "Explain attention mechanism"]
            return response, suggestions, metadata

        if re.search(r'\b(bye|goodbye|see\s+you)\b', cleaned_query):
            response = "Goodbye! 👋 Have a wonderful day ahead, and feel free to return whenever you need assistance."
            suggestions = ["Start new chat", "View pricing", "Help with account"]
            return response, suggestions, metadata

        # If raw output is clean and confident, return polished version
        if not is_truncated and len(detokenized_raw.split()) >= 4:
            suggestions = ["Can you explain more?", "What do you recommend?", "Contact support"]
            return detokenized_raw, suggestions, metadata

        # Intelligent helpful fallback
        fallback_response = f"I hear you regarding: *\"{user_query}\"*. Here is how I can assist you right now:\n\n" \
                            f"• **Account & Security**: Step-by-step guidance for password resets, unlocking accounts, and profile updates.\n" \
                            f"• **Product & Plan Recommendations**: Personalized advice comparing Starter ($10), Pro ($29), and Enterprise ($99) plans.\n" \
                            f"• **Technical Troubleshooting**: Crash diagnostic checklists and error resolution.\n" \
                            f"• **AI Architecture Deep Dive**: Detailed explanation of our Bidirectional LSTM and Bahdanau Attention layer."
        suggestions = ["Recommend a plan", "How do I reset my password?", "Troubleshoot app error", "Explain attention mechanism"]
        metadata["is_enriched"] = True
        return fallback_response, suggestions, metadata
