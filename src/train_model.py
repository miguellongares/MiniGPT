import argparse
from transformer import Decoder
import torch
from torch.nn import functional as F
from helpers import load_txt, load_encoder_decoder, create_batches, train_val_split
import os

#GPT-Configurations
emb_dim = 128*2
attention_dim = emb_dim #vanilla GPT where attention_dim == emb_dim 
text_length = 64        #how much context will the transformer take into acount
n_heads = 16*2          #number of heads in each multi-head transformer
n_layers = 6            #number of decoder layers 

#Train variables:
lr= 0.001
batch_size= 64
epochs= 5


def train_model(text_file, model_name):
    print('Loading the text file', text_file)

    #Load train data and encoder and set the token_dic:
    text = load_txt(text_file)
    encoder, _ = load_encoder_decoder(text)
    data = encoder(text)
    token_dic = len(set(data))

    #Create the model with the GPT-configurations
    model = Decoder(
    token_dic,
    emb_dim,
    attention_dim,
    n_heads,
    text_length,
    n_layers)

    #Train the model:
    train_data, val_data = train_val_split(data, 0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    val_batches = [create_batches(val_data, batch_size, text_length) for _ in range(80)]

    #Train loop:
    for iter in range(1001):
        optimizer.zero_grad()
        x, y = create_batches(train_data, batch_size, text_length)
        output = model(x)
        input = output.view((-1, token_dic))
        target = y.view(-1)
        loss = F.cross_entropy(input, target)
        if iter % 100 == 0:
            model.eval()
            with torch.no_grad():
                vals = []
                for x_val, y_val in val_batches:
                    val_logits = model(x_val).view(-1, token_dic)
                    vals.append(F.cross_entropy(val_logits, y_val.view(-1)).item())
                val_loss = sum(vals)/len(vals)
            model.train()
            print(f"Train {loss.item():.4f} | Val {val_loss:.4f}")
        loss.backward()
        optimizer.step()

    #Save the model with the pased name:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', 'saved_models', model_name+'.pt')
    PATH = os.path.abspath(path)
    torch.save(model, PATH)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer model with a certain text file and save the model')
    parser.add_argument('text_file', type=str, help='Name of the textfile stored in data folder')
    parser.add_argument('model_name', type=str, help='Name of the trained model stored in saved_models folder')
    args = parser.parse_args()

    train_model(args.text_file, args.model_name)

    #train_model('Don_Quijote_esp.txt', 'gpt_esp')