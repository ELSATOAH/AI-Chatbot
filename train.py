import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Laden der Artikeldaten

with open("data/articles.json", "r", encoding='utf-8') as file:
    articles = json.load(file)

# Erstellen von Trainingsdaten für das GPT-2-Modell

# Erstellen von Eingabe- und Ausgabepaaren basierend auf Artikeldaten
train_texts = []
for product in articles:
    context = (
        f"Produktname: {product['Description']}\n"
        f"Preis: {product['Unit Cost']} Euro\n"
        f"Farbe: {product['Colour']}\n"
        f"Beschreibung: {product.get('Description 2', 'Keine zusätzliche Beschreibung')}\n\n"
        f"Kunde: Wie viel kostet die {product['Description']}?\n"
        f"Antwort: Die {product['Description']} kostet {product['Unit Cost']} Euro.\n"
        f"###\n"
    )
    train_texts.append(context)

# Tokenisierung der Trainingsdaten

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 hat kein Standard-Pad-Token

train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Erstellen des Datasets

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.input_ids[idx])  # Für die Sprachmodellierung sind die Labels die input_ids
        }

dataset = CustomDataset(train_encodings)

#  Laden des GPT-2-Modells und Festlegen der Trainingsargumente

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir='./gpt2_model',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    logging_dir='./logs',
)

# Initialisieren des Trainers und Starten des Trainings

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Speichern des feinabgestimmten Modells und Tokenizers

trainer.save_model("./gpt2_model")
tokenizer.save_pretrained("./gpt2_model")

print("Feinabgestimmtes GPT-2-Modell und Tokenizer wurden gespeichert.")

# Erstellen des FAISS-Index

# Initialisieren des Sentence-Transformer-Modells
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Extrahieren der Produktbeschreibungen
descriptions = [item["Description"] for item in articles]

# Erzeugen der Vektoren für die Produktbeschreibungen
vectors = embedder.encode(descriptions)

# Ermitteln der Dimension der Vektoren
dimension = vectors.shape[1]

# Erstellen des FAISS-Index
index = faiss.IndexFlatL2(dimension)

# Hinzufügen der Vektoren zum Index
index.add(np.array(vectors))

# Speichern des FAISS-Index
faiss.write_index(index, "faissIndex.index")
print("FAISS-Index wurde gespeichert.")

# Speichern des Sentence-Transformer-Modells
embedder.save("sentence_transformer_model")
print("Sentence Transformer Modell wurde gespeichert.")

