from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as gensim
import random
import os




# For cleaning data to only nessecary info. 
# Specify index in current data file for word1, word2, and score
# And speciy separator used in index. 
# Will produce new data of the format: "word1   word   score"
# With tab being the separator.
def data_clean(filename, wrd1_idx, wrd2_idx, scr_idx):
    new_lines = []
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            split_line = line.strip().split()
            wrd1 = split_line[wrd1_idx]
            wrd2 = split_line[wrd2_idx]
            scr = split_line[scr_idx]

            new_lines.append(f"{wrd1}\t{wrd2}\t{scr}\n")

    with open(filename, 'w') as f:
        f.writelines(new_lines)

# Function for splitting golden standard data, into training(80%) and test (20%)
def split_data(filename):
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.read().splitlines()
        random.shuffle(lines)

        split = int(0.8 * len(lines))
        train_data = lines[:split]
        test_data = lines[split:]
    
    train_file = filename[:-4] + "_train.txt"
    print(train_file)
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    test_file = filename[:-4] + "_test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + '\n')

# Counts and findsall unknown words of text files in folder
def count_unknowns(path):
    file_names = os.listdir(path)
    words = set()

    # Get all unique words
    for filename in file_names:
        with open(os.path.join(path, filename), "r", encoding='utf-8') as f:   
            lines = f.read().splitlines()
            for line in lines:
                word = line.strip()
                words.add(word)

    # Word2Vec is only nesssecary, otherwise takes forever 
    models = {
        "Word2Vec"  : gensim.load("word2vec-google-news-300"),
    }

    unknowns = set()
    for word in words:
        for model_name, model in models.items():
            try:
                if isinstance(model, KeyedVectors):
                    if word not in model:
                        unknowns.add(word)
                        break  # No need to check other models
                elif isinstance(model, SentenceTransformer):
                    _ = model.encode([word])  # Try encoding
            except: # If encoding fails
                unknowns.add(word)
                break 

    print(f"Words not known by at least one model: {len(unknowns)}")
    print(unknowns)


# Removes unknown words from text files in folder
# !!! BE CAREFUL IT RECURSIVELY SEARCHES FOLDER !!!
def remove_unknowns(path, unknown_words):
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            f_path = os.path.join(path, filename)
            with open(f_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            new_lines = []
            unknowns = []
            for line in lines:
                word = line.strip().lower()
                if word not in unknown_words:
                    new_lines.append(word)
                else:
                    unknowns.append(word)
            
            if len(new_lines) < len(lines):
                print(f"{filename} : unknown words count: {len(unknowns)} unknown words removed: {unknowns}")
                with open(f_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines))
        else: # To remove unknowns from similar folder BE CAREFUL IS RECURSIVE
            new_path = os.path.join(path, filename)
            remove_unknowns(new_path, unknowns)






