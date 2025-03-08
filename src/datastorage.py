from scipy.sparse import load_npz
from scipy.sparse import save_npz
import pandas as pd
import time
import re



def load_stopwords(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()
  
def load_data(filename):
    chunks = pd.read_csv(
        filename, 
        encoding='utf-8', 
        usecols=["lyrics"],
        chunksize=10
        )
    return chunks

def load_sparse_matrix_lil(filename="co_occurrence_matrix.npz"):
    co_matrix_csr = load_npz(filename)
    co_matrix_lil = co_matrix_csr.tolil()
    print(f"Loader sparse matrix from {filename}")
    return co_matrix_lil

def save_sparse_matrix_npz(co_matrix, filename="co_occurrence_matrix.npz"):
    co_matrix_csr = co_matrix.tocsr()  
    save_npz(filename, co_matrix_csr)
    print(f"Saved sparse matrix to {filename}")