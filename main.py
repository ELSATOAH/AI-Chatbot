#import cohere
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GPT2Tokenizer
import json
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Setze einen geheimen Schlüssel für CSRF-Schutz

# Laden der FAISS-Index und Sentence-Transformer-Modell

# Laden des FAISS-Index
index = faiss.read_index("faissIndex.index")

# Laden des Sentence-Transformer-Modells
embedder = SentenceTransformer("sentence_transformer_model")

# Laden des feinabgestimmten GPT-2-Modells

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_model")
tokenizer.pad_token = tokenizer.eos_token

generator = pipeline("text-generation", model="./gpt2_model", tokenizer=tokenizer)

# Laden der Artikeldaten

with open("data/articles.json", "r", encoding='utf-8') as file:
    articles = json.load(file)

# Funktionen zur semantischen Suche und Textgenerierung

def findBestMatch(query, k=3):
    """Funktion zur semantischen Suche im FAISS-Index."""
    query_vector = embedder.encode([query])
    _, indices = index.search(query_vector, k)
    return [articles[i] for i in indices[0]]

def generateText(prompt):
    """Generiert eine Antwort basierend auf der Kundenanfrage."""
    products = findBestMatch(prompt, k=3)  # Hole die Top 3 passenden Produkte
    
    context = ""
    for product in products:
        context += (
            f"Produktname: {product['Description']}\n"
            f"Preis: {product['Unit Cost']} Euro\n"
            f"Farbe: {product['Colour']}\n"
            f"Beschreibung: {product.get('Description 2', 'Keine zusätzliche Beschreibung')}\n\n"
        )
    context += f"Kunde: {prompt}\nAntwort:"
    
    # Generieren der Antwort mit dem feinabgestimmten Modell
    response = generator(
        context,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]
    
    # Extrahieren der Antwort nach dem "Antwort:"-Token
    if "Antwort:" in response:
        generated_answer = response.split("Antwort:")[1].strip()
    else:
        generated_answer = response.strip()
    
    return generated_answer

# Flask-Formular für die Benutzereingabe

class Form(FlaskForm):
    text = StringField('Stelle eine Frage zu unseren Produkten:', validators=[DataRequired()])
    submit = SubmitField('Absenden')

# **Schritt 6: Flask-Routen und -Ansichten**

@app.route('/', methods=['GET', 'POST'])
def home():
    form = Form()
    antwort = None  # Standardwert setzen
    
    if form.validate_on_submit():
        text = form.text.data  # Text aus der Benutzereingabe
        # Generiere eine Antwort basierend auf der Benutzereingabe
        antwort = generateText(text)
    
    return render_template('Homepage.html', form=form, output=antwort)

if __name__ == "__main__":
    app.run(debug=True)


   # co = cohere.Client('i6GoOZjxHDlZE9HIP9d1GtDyw4Rcg4r9Myw3VMdP')

   #if form.validate_on_submit():
   #    text = form.text.data
   #    response = co.generate(
   #        model='command-nightly',
   #        prompt=text,
   #        max_tokens=300,
   #        temperature=0.9,
   #        k=0,
   #        p=0.75,
   #        stop_sequences=[],
   #        return_likelihoods='NONE'
   #    )
   #    output = response.generations[0].text
   #    return render_template('Homepage.html', form=form, output=output)

   #return render_template('Homepage.html', form=form, output=None)


if __name__ == "__main__":
    app.run(debug=True)
