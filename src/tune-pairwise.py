from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.losses import ContrastiveLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import random


# sInherits from torch.nn.Module, the base class neural network modules in PyTorch
class RelativeRelatednessLoss(nn.Module):
    def __init__(self, model, margin=0.05):
        super().__init__()
        self.model = model
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1) # cosine similarity, comparing vectors element-wise

    def forward(self, sentence_features, labels=None):
        # get all the embeddings
        a1 = self.model(sentence_features[0])['sentence_embedding']
        b1 = self.model(sentence_features[1])['sentence_embedding']
        a2 = self.model(sentence_features[2])['sentence_embedding']
        b2 = self.model(sentence_features[3])['sentence_embedding']

        rel1 = self.cos(a1, b1) # Cosine similarity between first pair of  words
        rel2 = self.cos(a2, b2) # Cosine similarity between second pair of words

        loss = F.relu(self.margin + rel2 - rel1).mean()
        return loss
    

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
    

# Convert similarity dictionary into relative pairs (a1, b1, a2, b2)
# where rel(a1, b1) > rel(a2, b2)
def generate_relative_pairs(sim_dict, max_pairs=None):
    items = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    relative_pairs = []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (a1, b1), rel1 = items[i]
            (a2, b2), rel2 = items[j]

            if rel1 > rel2:  # ordered
                relative_pairs.append((a1, b1, a2, b2))
    return relative_pairs


# Normalize all scores in dataset
def normalize(dic, max_val):
    return {key: val / max_val for key, val in dic.items()}

def prepare_data(data_dict):
    return [InputExample(texts=[w1, w2], label=score) for (w1, w2), score in data_dict.items()]

def prepare_rrl_data(relative_pairs):
    input_examples = []
    for a1, b1, a2, b2 in relative_pairs:
        input_examples.append(InputExample(texts=[a1, b1, a2, b2]))
    return input_examples
   
def finetune(model, loss, train_data, output_path):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"Model fine-tuned, saved to {output_path}")


data_path = "data/relatedness/"
data_name = "WordSim353-REL"
data_max = 10
model_name = "MPNet"
model = SentenceTransformer("all-mpnet-base-v2")
loss = RelativeRelatednessLoss(model)
#loss = CosineSimilarityLoss(model)
#loss = ContrastiveLoss(model)

output_path = "data/models/"

data = normalize((load_standard(data_path + data_name + "/train.txt")), data_max)
if isinstance(loss, (CosineSimilarityLoss, ContrastiveLoss)):
    train_data = prepare_data(data)
    if isinstance(loss, CosineSimilarityLoss): loss_name = "cossim"
    else: loss_name = "contra"
elif isinstance(loss, RelativeRelatednessLoss):
    train_data = generate_relative_pairs(data)
    split = round(len(train_data) * 0.025)
    random.shuffle(train_data)
    train_data = train_data[:split]
    train_data = prepare_rrl_data(train_data)
    loss_name = "rrl"

name = data_name+"-"+loss_name+"-t-"+model_name
finetune(model, loss, train_data, output_path+name)

