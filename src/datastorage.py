from scipy.sparse import load_npz
from scipy.sparse import save_npz
import pandas as pd
import os 


def load_stopwords(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()
  
def load_lyrics_data(filename):
    chunks = pd.read_csv(
        filename, 
        encoding='utf-8', 
        usecols=["lyrics"],
        chunksize=15
        )
    return chunks

def load_book_data(filename, chunk_size=15):
    with open(filename, "r", encoding="utf-8") as file:
        while True:
            lines = [file.readline().strip() for _ in range(chunk_size)]
            if not lines or all(line == "" for line in lines):  # Stop if no more lines
                break
            yield lines

   
