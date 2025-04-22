import random
import pandas as pd


# For cleaning data to only nessecary info. 
# Specify index in current data file for word1, word2, and score
# And speciy separator used in index. 
# Will produce new data of the format: "word1   tword   score"
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

# For Loading Golden Standard Datasets
def load_standard(filename):
    dic = {}
    with open(filename, "r", encoding='utf-8') as f:   
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Skipping malformed line {i + 1}: {line} ----------------------------------------")
                continue
            w1, w2, scr = parts
            key = tuple(sorted([w1, w2]))   
            dic[key] = float(scr)
    return dic

# Function for loading words from categories



        
# Function for loading files with list of words, used for stop words, and top 10000 english words
def load_words(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()

def get_categories_words(categories):
    stop_words = load_words("data/stop_words_english.txt")  # Stop words to ignore
    words = []

    for category in categories:
        words.extend(load_words(f"data/categories/{category}"))
    
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def get_top_words():
    top_words = load_words("data/google-10000-english.txt") # Top 10000 english words
    stop_words = load_words("data/stop_words_english.txt")  # Stop words to ignore

    # Filter stop words from top 10000 words
    filtered_words = [word for word in top_words if word not in stop_words]
    return filtered_words
    

def save_words_df(df):
    df.to_csv("data/word_similarity.csv", index=False)

def load_words_df():
    return pd.read_csv("data/word_similarity.csv")



