import torch
import torch.nn as nn
from torch.nn import functional as F
from helpers import load_txt, load_encoder_decoder, create_batches, train_val_split

#Input shape consists of:
#   -B: BATCH Parallel samples, Batch size
#   -T: Text Length of the text
#   -C: Chanels Embedding dimension

#   -AD: Attention dimension

#Creating a vanilla transformer model where:
#   head_size = embedding_dimension // head_n
#IT IS POSSIBLE TO CHOOSE OTHER DIMENSIONS BUT USSING A FINAL LINEAR LAYER TO MATCH VECTOR

class Embedding(nn.Module):
    def __init__(self, token_dic, emb_dim, text_length):
        super().__init__()
        #input shape (B, T)
        self.embedding_table = nn.Embedding(token_dic, emb_dim) #(B,T,C)
        self.possitional_emb = nn.Embedding(text_length, emb_dim)

    def forward(self, x):
        out = self.embedding_table(x) + self.possitional_emb(torch.arange(x.shape[1]))
        return out #Shape(B, text_lengt, emb_dim)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, att_dim, n_heads, text_length, masked = False):
        super().__init__()
        assert att_dim % n_heads == 0, "att_dim must be divisible by n_heads"
        self.att_dim = att_dim
        self.n_heads = n_heads
        self.head_dim = att_dim // n_heads

        self.attentionLayer = nn.Linear(emb_dim, att_dim * 3)
        if masked:
            self.register_buffer('mask', torch.tril(torch.ones(text_length, text_length)))
        else:
            self.register_buffer('mask', torch.ones(text_length, text_length))

    def forward(self, x):
        B, T, _ = x.shape

        # project to Q, K, V
        Q, K, V = torch.split(self.attentionLayer(x), self.att_dim, dim=-1)  # (B,T,att_dim)

        # reshape into heads: (B, n_heads, T, head_dim)
        q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scale = self.head_dim ** 0.5
        qk = (q @ k.transpose(-1, -2)) / scale            # (B, n_heads, T, T)
        qk = qk.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(qk, dim=-1)

        out = att @ v                                     # (B, n_heads, T, head_dim)

        # merge heads: (B, T, att_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.att_dim)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, att_dim, n_heads, text_length):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(emb_dim, att_dim, n_heads, text_length, masked=True)
        self.FeedForward = nn.Sequential(nn.Linear(att_dim, 4*att_dim),
                                         nn.ReLU(),
                                         nn.Linear(4*att_dim, att_dim),
                                         nn.ReLU())
        self.ln1 = nn.LayerNorm(att_dim)
        self.ln2 = nn.LayerNorm(att_dim)
        
    def forward(self, x):
        #First part of masked multi-head-attention with res connection and layser norm
        mask_att = self.masked_multi_head_attention(x)
        att = self.ln1(mask_att) + x
        #Third part using Feed Forward with res connection and layer norm
        feedForward_x = self.FeedForward(att)
        out = self.ln2(feedForward_x) + att
        return out #Shape(B, T, AD == C)

class Decoder(nn.Module):
    def __init__(self, token_dic, emb_dim, attention_dim, n_heads, text_length):
        super().__init__()

        self.embedding = Embedding(token_dic, emb_dim, text_length)
        #self.multihead = MultiHeadAttention(emb_dim, attention_dim, n_heads, text_length)
        self.transformerblock = TransformerBlock(emb_dim, attention_dim, n_heads, text_length)
        self.ln = nn.Linear(attention_dim, token_dic)

    def forward(self, x):
        emb_x = self.embedding(x) #Shape(B,T,C)
        att_x = self.transformerblock(emb_x) #Shape(B,T,AD)
        logits = self.ln(att_x)

        return logits
    
    def generate(self, context, generation_lenght):
        text_idx = context
        for i in range(generation_lenght):
            input = text_idx[:,-text_length:]
            out = self(input) 
            next_x = F.softmax(out[:,-1,:], dim=-1) #Shape(B, C) last character
            next_idx = torch.multinomial(next_x, num_samples=1) # (B,1)
            text_idx = torch.cat([text_idx,next_idx], dim=1)
            
        return text_idx
        

####Run script#####
    
text = load_txt('Don_Quijote_esp.txt')
encoder, decoder = load_encoder_decoder(text)
data = encoder(text)
token_dic = len(set(data))
emb_dim = 128
text_length = 64
attention_dim = 128
n_heads = 16

train_data, val_data = train_val_split(data, 0.9)

model = Decoder(token_dic, emb_dim, attention_dim, n_heads, text_length)
optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-3)

val_batches = [create_batches(val_data, n_batches=64, length=text_length) for _ in range(80)] ##test

#train loop:
for iter in range(2000):
    optimizer.zero_grad()
    x, y = create_batches(train_data, n_batches=64, length= text_length) #shape (B, T), (B, T)
    output = model(x) #output shape (B, T, T)
    input = output.view((-1,token_dic))
    target = y.view(-1)
    loss = F.cross_entropy(input, target) #has to be (B*T, C) and (B*T)
    if iter % 100 == 0:
        model.eval()
        with torch.no_grad():
            vals = []
            for x_val, y_val in val_batches:
                val_logits = model(x_val).view(-1, token_dic)
                vals.append(F.cross_entropy(val_logits, y_val.view(-1)).item())
            val_loss = sum(vals) / len(vals)
        model.train()
        print(f"Train {loss.item():.4f} | Val {val_loss:.4f}")
    loss.backward()
    optimizer.step()#


gen = model.generate(torch.zeros((1,1), dtype=torch.long), generation_lenght= 1000)
print(''.join(decoder(gen[0].tolist())))
