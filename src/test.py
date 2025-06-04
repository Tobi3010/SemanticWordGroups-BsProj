from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as api
import os

def count_not_in_models(path):
    file_names = os.listdir(path)
    words = set()

    # Gather all unique words
    for filename in file_names:
        with open(os.path.join(path, filename), "r", encoding='utf-8') as f:   
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                word = line.strip().lower()
                words.add(word)
                

    # Load all models
    models = {
        "Word2Vec"  : api.load("word2vec-google-news-300"),
    }

    unknown_words = set()
    
    for word in words:
        for model_name, model in models.items():
            try:
                if isinstance(model, KeyedVectors):
                    if word not in model:
                        unknown_words.add(word)
                        break  # No need to check others
                elif isinstance(model, SentenceTransformer):
                    _ = model.encode([word])  # Try encoding to test for error
            except:
                unknown_words.add(word)
                break  # If encoding fails, consider it unknown

    print(f"Words not known by at least one model: {len(unknown_words)}")
    print(unknown_words)



def remove_unknowns(path, unknown_words):
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            full_path = os.path.join(path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            filtered_lines = []
            unknowns = []
            for line in lines:
                word = line.strip().lower()
                if word not in unknown_words:
                    filtered_lines.append(word)
                else:
                    unknowns.append(word)
            
            if len(filtered_lines) < len(lines):
                print(f"{filename} : unknown words count: {len(unknowns)} unknown words removed: {unknowns}")
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(filtered_lines))
        else:
            new_path = os.path.join(path, filename)
            remove_unknowns(new_path, unknown_words)

        


unknown_words = {'slovene', 'turkmen', 'hakka', 'oromo', 'assamese', 'lithuania', 'montenegro', 'estonian', 'uranus', 'ossetian', 'bhutan', 'amharic', 'grey', 'tuvalu', 'smoot', 'tsonga', 'cambodia', 'uzbek', 'croatian', 'slovak', 'tahitian', 'botswana', 'bhojpuri', 'rudolph', 'micronesia', 'bolivia', 'kyrgyz', 'bosnian', 'mayan', 'mauritius', 'doughnut', 'brunei', 'tajik', 'guyana', 'javanese', 'eritrea', 'frisian', 'latvia', 'gutenberg', 'samoa', 'mozambique', 'to', 'bulgarian', 'madagascar', 'vanuatu', 'seychelles', 'latvian', 'newfoundland', 'nicaragua', 'xhosa', 'sunda', 'fermi', 'albanian', 'khmer', 'pashto', 'armenia', 'nauru', 'azerbaijan', 'druze', 'yiddish', 'senegal', 'tajikistan', 'infanta', 'belarusian', 'malawi', 'namibia', 'albania', 'quechua', 'slovenia', 'kazakhstan', 'malagasy', 'benin', 'infante', 'gabon', 'abkhaz', 'oriya', 'sindhi', 'guatemala', 'palau', 'slovakia', 'afrikaans', 'lesotho', 'andorra', 'cameroon', 'swahili', 'comoros', 'burundi', 'macedonian', 'hmong', 'archduchess', 'zhuang', 'luxembourgish', 'archaeology', 'axe', 'kazakh', 'belarus', 'gujarati', 'tanzania', 'mongolia', 'swaziland'}
remove_unknowns("data/categories/all2", unknown_words)

#count_not_in_models()