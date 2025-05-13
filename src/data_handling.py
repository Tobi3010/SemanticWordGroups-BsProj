import os
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

        
# Function for loading files with list of words, used for stop words, and top 10000 english words
def load_words(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()

def get_categories_words(categories, amount=0):
    stop_words = load_words("data/stop_words_english.txt")  # Stop words to ignorez
    words = []

    for category in categories:
        if amount == 0:
            words_to_add = load_words(f"data/categories/{category}")
        else:
            words_to_add = load_words(f"data/categories/{category}")
            random.shuffle(words_to_add)
            words_to_add = words_to_add[:amount]
        words.extend(words_to_add)
        words.append(category.split("-")[0])
    
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def make_word_category_df(categories, include_category_word=True):
    data = {}

    for category, n_words in categories.items():
        words_to_add = load_words(f"data/categories/{category}")
        if n_words > 0:
            random.shuffle(words_to_add)
            words_to_add = words_to_add[:n_words]
        category_word = category.split("-")[0]
        
        if include_category_word and category_word not in words_to_add:
            words_to_add.append(category_word)
        
        data[category_word] = words_to_add

    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    return df

def save_words_category_df(categories, df):
    # Convert wide format to long format: word, category
    df_melted = df.melt(var_name="category", value_name="word").dropna()

    # Format folder name
    parts = []
    for category, n_words in categories.items():
        category_name = category.split("-")[0]
        parts.append(f"{category_name}{n_words}")
    name = "-".join(parts)
    print(name)

    # Create directory and save the melted DataFrame
    directory_name = f"data/networks/{name}"
    os.makedirs(directory_name, exist_ok=True)
    df_melted[["word", "category"]].to_csv(f"{directory_name}/categories_words.csv", index=False)
    
def load_df_category_words(name):
    return pd.read_csv(f"data/networks/{name}/categories_words.csv")

def load_df_words(name):
    df = pd.read_csv(f"data/networks/{name}/categories_words.csv")
    return df["word"].dropna().tolist()

def save_df_cossim(df, name, model_name):
    df.to_csv(f"data/networks/{name}/{model_name}.csv", index=False)

def load_df_cossim(name, model_name):
    return pd.read_csv(f"data/networks/{name}/{model_name}.csv")




