from sentence_transformers import SentenceTransformer, util, models, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.losses import CosineSimilarityLoss
from data_handling import load_standard



def fine_tune_model(model_path, data_path, n, output_path):
    model = SentenceTransformer(model_path)
    data = load_standard(data_path)
    train_data = [
        InputExample(texts=[word1, word2], label=score / n)
        for (word1, word2), score in data.items()
    ]

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )

    print(f"Model saved to {output_path}")


fine_tune_model("all-roberta-large-v1", "data/similarity/SimVerb-3500.txt", 10, "data/models/SimVerb-3500-tuned-Roberta")



