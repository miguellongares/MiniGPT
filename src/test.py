import os
from helpers import load_txt

file = 'input.txt'

text = load_txt(file)

print(f"Loaded {len(text)} characters from {file}")