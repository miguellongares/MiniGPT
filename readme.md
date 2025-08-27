# Vanilla Transformer (GPT-style) in PyTorch

This project implements a **decoder-only Transformer model** (similar to a simplified GPT) from scratch in PyTorch.  
It is designed to train on raw text and generate new sequences autoregressively.

---

## Features
- Token + positional embeddings with dropout regularization.
- Masked multi-head self-attention with dropout:
  - Dropout on attention weights (prevents over-reliance on specific token-to-token connections).
  - Dropout on output projection (prevents co-adaptation of heads).
- Feedforward network with dropout in the hidden layer (reduces memorization).
- Residual connections with dropout (standard in transformers).
- Configurable number of layers, heads, embedding size, and context length.
- Text generation via sampling from the trained model.

---

##  Model Architecture
- **Embedding Layer**: Combines token embeddings and positional embeddings.  
- **Multi-Head Attention**: Computes masked self-attention over a context window.  
- **Transformer Block**:  
  - LayerNorm → Masked Multi-Head Attention → Residual + Dropout  
  - LayerNorm → FeedForward → Residual + Dropout  
- **Decoder**: Stacks multiple Transformer blocks and projects back to vocabulary logits.

---

##  Configuration
Default training configuration in the script:
- `emb_dim = 256` (embedding size)  
- `attention_dim = emb_dim` (attention dimension = embedding dim)  
- `text_length = 64` (context window size)  
- `n_heads = 32` (number of attention heads)  
- `n_layers = 6` (number of transformer blocks)  
- Optimizer: `AdamW(lr=1e-3)`  
- Loss: Cross-entropy over token predictions  

---

##  Training Data
The example script can be loaded with **Don Quijote (Spanish text)** or with **Romeo & Juliet (English text)**:  
```python
text = load_txt('Don_Quijote_esp.txt')
text = load_txt('Romeo_Juliet_ENG.txt')