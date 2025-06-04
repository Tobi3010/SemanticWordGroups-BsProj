from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader
import os
import random

def load_grouped_words(folder):
    data = {}
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            label = fname.rsplit(".", 1)[0]
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]
                if len(words) > 1:
                    data[label] = words
    return data

def make_triplets(data, num_triplets=3000):
    triplets = []
    labels = list(data.keys())
    for _ in range(num_triplets):
        pos_label = random.choice(labels)
        neg_label = random.choice([l for l in labels if l != pos_label])
        anchor, positive = random.sample(data[pos_label], 2)
        negative = random.choice(data[neg_label])
        triplets.append((anchor, positive, negative))
    return triplets


# Load triplets
data = load_grouped_words("data/categories/all")
triplets = make_triplets(data)

# Convert to InputExample
examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]

# Model
model = SentenceTransformer("data/models/MEN-cossim-t-MPNet")

# DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)

# Loss function
train_loss = losses.TripletLoss(model=model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True,
    output_path='data/models/cluster'
)