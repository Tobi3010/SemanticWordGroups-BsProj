import scipy.sparse
from csv import DictReader

def load_stopwords(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()
  

def load_data(filename):
    with open(filename, newline='', encoding='utf-8') as file:
        data = DictReader(file)  # Automatically uses first row as headers
        lyrics_list = [row for row in data]  # Convert iterator to list of dictionaries

    return lyrics_list

def load_sparse_matrix_lil(filename="co_occurrence_matrix.npz"):
    co_matrix_csr = scipy.sparse.load_npz(filename)
    co_matrix_lil = co_matrix_csr.tolil()
    print(f"Loader sparse matrix from {filename}")
    return co_matrix_lil

def save_sparse_matrix_npz(co_matrix, filename="co_occurrence_matrix.npz"):
    co_matrix_csr = co_matrix.tocsr()  
    scipy.sparse.save_npz(filename, co_matrix_csr)
    print(f"Saved sparse matrix to {filename}")