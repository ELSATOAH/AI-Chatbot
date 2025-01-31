from transformers import Trainer, TrainingArguments, AutoModelForCasualLM, AutoTokenizer
import datasets

modelName= "gpt2"
tokenizer = AutoTokenizer.from_pretrained(modelName)

# Trainingsdaten
data= datasets.Dataset.from_dict({
    "text": ["Hallo! Wie kann ich dir helfen?", "Mir geht es gut, danke der Nachfrage!", "Ich kann einfache Fragen beantworten.", "Auf Wiedersehen!"]
})

# Laden von großen Daten (z.B. aus einer CSV-Datei)
#dataset = datasets.load_dataset('csv', data_files='path_to_your_large_dataset.csv')

# Tokenisierung optimiert für große Datasets
#def tokenize_function(examples):
#    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Tokenisieren des gesamten Datasets
#tokenized_data = dataset.map(tokenize_function, batched=True, num_proc=4)


# Tokenisierung
tokenizedData = data.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

# Modell
model = AutoModelForCasualLM.from_pretrained(modelName)

# Trainingsparameter
TrainingArguments = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_total_limit=10,

    #logging_dir='./logs',
    #logging_steps=10,
    #evaluation_strategy="epoch",  # Für regelmäßige Evaluation während des Trainings
    #weight_decay=0.01,
    #save_strategy="epoch",  # Speichern nach jeder Epoche
    #load_best_model_at_end=True,


)

# Trainer
trainer= Trainer(
    model=model,
    args=TrainingArguments,
    train_dataset=tokenizedData["train"],
)

# Training
trainer.train()

# Speichern
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
