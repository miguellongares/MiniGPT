import torch
import os
import argparse
from transformer import Decoder
from helpers import load_txt, load_encoder_decoder
from config import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('model_file', type=str, help='Define which model.pt will be loaded')
    args = parser.parse_args()

    #Load the decoder
    text = load_txt('Don_Quijote_esp.txt')
    _, decoder = load_encoder_decoder(text)

    #Path to the trained model:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', 'saved_models', args.model_file)
    PATH = os.path.abspath(path)

    #Load model
    model = torch.load(PATH, weights_only=False)
    model.eval()

    #Generate text:
    text_generated = model.generate(torch.zeros((1,1), dtype=torch.long), generation_lenght= 1000)
    print(''.join(decoder(text_generated[0].tolist())))

