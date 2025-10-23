#GPT-Configurations
emb_dim = 128*2
attention_dim = emb_dim #vanilla GPT where attention_dim == emb_dim 
text_length = 64        #how much context will the transformer take into acount
n_heads = 16*2          #number of heads in each multi-head transformer
n_layers = 6            #number of decoder layers  