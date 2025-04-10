import random


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
        
def load_stopwords(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()
    


