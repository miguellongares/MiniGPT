import os
import torch

#Load text file inside the data folder
def load_txt(data_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', 'data', data_file)
    path = os.path.abspath(path)
    with open(path, 'r', encoding='utf-16') as f:
        text = f.read()
    return text

def load_encoder_decoder(text):
    '''
    Given a certain text it returns encoder and decoder functions
    '''
    vocab_list = sorted(list(set(text)))
    encoder_dic = {ch:i for i, ch in enumerate(vocab_list)}
    decoder_dic = {i:ch for i, ch in enumerate(vocab_list)}
    encoder = lambda text: [encoder_dic[ch] for ch in text]
    decoder = lambda code: [decoder_dic[i] for i in code]
    return encoder, decoder

def create_batches(text, n_batches, length):
    encoder, _ = load_encoder_decoder(text)
    start_idxs = torch.randint(0, len(text)-length, (n_batches,))
    X = torch.vstack([torch.tensor(encoder(text[idx:idx+length])) for idx in start_idxs])
    Y = torch.vstack([torch.tensor(encoder(text[idx+1:idx+length+1])) for idx in start_idxs])
    return X, Y

if __name__ == '__main__':
    text = load_txt('input.txt')
    print(create_batches(text, 3, 4))
    