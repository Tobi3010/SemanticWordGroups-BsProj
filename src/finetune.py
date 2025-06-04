from sentence_transformers import SentenceTransformer, util, models, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.losses import CosineSimilarityLoss
from data_handling import load_standard
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



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

# Function for fine tuning neural models from 'sentence_transformers'
def fine_tune_model_rrl(model, train_data, output_path):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = RelativeRelatednessLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )

    print(f"Model saved to {output_path}")


# Function for fine tuning neural models from 'sentence_transformers'
def fine_tune_model_cossim(model, train_data, output_path):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"Model saved to {output_path}")

def fine_tune_model_contra(model, train_data, output_path):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.ContrastiveLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"Model fine-tuned with ContrastiveLoss and saved to {output_path}")


# Function for evaluating neural models
def evaluate(model, standard):
    pred_scr = []
    true_scr = []
    for (wrd1, wrd2), scr in standard.items():
        emb1 = model.encode(wrd1, convert_to_tensor=True, device=device)
        emb2 = model.encode(wrd2, convert_to_tensor=True, device=device)
        cos_sim = util.cos_sim(emb1, emb2).item() 
        pred_scr.append(cos_sim)
        true_scr.append(scr)

    spearman, _ = spearmanr(pred_scr, true_scr)    
    return spearman


# Normalize all scores in dataset
def normalize(dic, max_val):
    return {key: val / max_val for key, val in dic.items()}

def prepare_train_data(data_dict):
    return [InputExample(texts=[w1, w2], label=score) for (w1, w2), score in data_dict.items()]

def prepare_train_data2(relative_pairs):
    """
    Converts (a1, b1, a2, b2) tuples into InputExamples
    for RelativeRelatednessLoss training.
    """
    input_examples = []
    for a1, b1, a2, b2 in relative_pairs:
        input_examples.append(InputExample(texts=[a1, b1, a2, b2]))
    return input_examples


# Path to semantic relatedness datasets
REL_data_path = "data/relatedness/"
# Dataset names, and their highest possible score (used for normalization)
REL_data = {"MEN" : 50}
# Load and normalize datasets for training
REL_train = {data : normalize(
    (load_standard(REL_data_path + data + "/train.txt")), REL_data[data]) for data in REL_data.keys()} 
# Load datasets for testing
REL_test = {data : load_standard(REL_data_path + data + "/test.txt") for data in REL_data.keys()}

model_path = "data/models/"
for train_name, train_data in REL_train.items():
    model = SentenceTransformer("data/models/SEMCAT-Triple-t-MPNet")
    model_name = train_name + "-cossim-t"
    output_path = model_path + model_name
    print(f"Fine tuning MPNet on {train_name} : STARTS")
    """
    train_data = generate_relative_pairs(train_data)
    split = round(len(train_data) * 0.01)
    print(f"data len : {split}")
    random.shuffle(train_data)
    train_data = train_data[:split] 
    train_data = prepare_train_data2(train_data)
    fine_tune_model_rrl(model, train_data, output_path)
    """
    train_data = prepare_train_data(train_data)
    fine_tune_model_cossim(model, train_data, output_path)
    print(f"Fine tuning MPNet on {train_name} : ENDED")
   
    """
    print(f"EVALUATING OF {model_name} : STARTS")
    for test_name, test_data in REL_test.items():
        spearman = evaluate(tuned_model, test_data)
        print(f"\t{test_name} Spearman Correlation : {spearman:.4f}")
    print(f"EVALUATING OF {model_name} : ENDED\n")
    """



