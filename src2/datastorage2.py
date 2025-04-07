
def load_wordsim(filename):
    dic = {}
    with open(filename, "r", encoding='utf-8') as f:   
        lines = f.read().splitlines()
        for line in lines:
            w1, w2, scr = line.split("\t")
            key = tuple(sorted([w1, w2]))   
            dic[key] = scr
    return dic
            
def load_stopwords(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()