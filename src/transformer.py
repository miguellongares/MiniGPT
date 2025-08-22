import torch
import torch.nn as nn
from torch.nn import functional as F
from helpers import load_txt, load_encoder_decoder, create_batches

#Input shape consists of:
#   -B: BATCH Parallel samples, Batch size
#   -T: Text Length of the text
#   -C: Chanels Embedding dimension

class Embedding(nn.Module):
    def __init__(self, token_dic, emb_dim, text_length):
        super().__init__()
        #input shape (B, T)
        self.embedding_table = nn.Embedding(token_dic, emb_dim) #(B,T,C)
        self.possitional_emb = nn.Embedding(text_length, emb_dim)

    def forward(self, x):
        out = self.embedding_table(x) + self.possitional_emb(torch.arange(x.shape[1]))
        return out #Shape(B, text_lengt, emb_dim)
    
class Decoder(nn.Module):
    def __init__(self, token_dic, emb_dim, text_length):
        super().__init__()

        self.embedding = Embedding(token_dic, emb_dim, text_length)
        self.ln = nn.Linear(emb_dim, token_dic)

    def forward(self, x):
        emb_x = self.embedding(x) #Shape(B,T,C)
        logits = self.ln(emb_x)

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
        

    
text = load_txt('input.txt')
token_dic = len(set(text))
emb_dim = 32
text_length = 10

model = Decoder(token_dic, emb_dim, text_length)
optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-2)

#train loop:
for iter in range(1000):
    optimizer.zero_grad()
    x, y = create_batches(text, n_batches=32, length= text_length) #shape (B, T), (B, T)
    output = model(x) #output shape (B, T, T)
    input = output.view((-1,token_dic))
    target = y.view(-1)
    loss = F.cross_entropy(input, target) #has to be (B*T, C) and (B*T)
    if (iter%100 == 0): print('loss is: ', loss.item())
    loss.backward()
    optimizer.step()#
    
gen = model.generate(torch.zeros((1,1), dtype=torch.long), generation_lenght= 1000)
encoder, decoder = load_encoder_decoder(text=text)
print(''.join(decoder(gen[0].tolist())))




