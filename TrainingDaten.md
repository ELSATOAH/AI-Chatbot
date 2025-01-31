 input_text,output_text
"Wie geht es dir?","Mir geht es gut, danke der Nachfrage!"
"Was kannst du?","Ich kann einfache Fragen beantworten."



---

### Struktur einer CSV für Produktdaten

Wenn du einen Chatbot mit Produktdaten trainieren möchtest, solltest du die **CSV-Datei** so strukturieren, dass die relevanten Informationen über die Produkte in einer gut lesbaren und leicht zugänglichen Form vorliegen. Dabei wäre es hilfreich, die Spalten nach den verschiedenen Merkmalen der Produkte zu gliedern und gegebenenfalls auch Links zu den Produktseiten hinzuzufügen.

#### Beispielhafte Struktur einer CSV für Produktdaten:

| Produktname        | Beschreibung                                    | Preis | Link                           | Kategorie    |
|--------------------|-------------------------------------------------|-------|--------------------------------|--------------|
| Fahrrad X1         | Ein leichtes, sportliches Fahrrad für Anfänger  | 499   | https://example.com/fahrrad-x1 | Fahrräder    |
| Fahrrad X2         | Ein robustes Mountainbike für anspruchsvolle Fahrten | 799   | https://example.com/fahrrad-x2 | Mountainbikes |
| Helm Y             | Ein sicherer Helm für Fahrradtouren              | 99    | https://example.com/helm-y     | Zubehör      |
| Fahrradschloss Z   | Ein langlebiges Fahrradschloss mit hohem Sicherheitsniveau | 39    | https://example.com/fahrradschloss-z | Zubehör   |

### Wichtige Überlegungen:

1. **Produktname**: Der Name des Produkts, der im Chatbot verwendet wird, um auf das Produkt zu verweisen.
2. **Beschreibung**: Eine kurze Beschreibung des Produkts, die der Chatbot verwenden kann, um auf die Merkmale und Vorteile des Produkts einzugehen.
3. **Preis**: Der Preis des Produkts, den der Chatbot in den Antworten einfügen kann, wenn er nach Preisen gefragt wird.
4. **Link**: Ein Link zum Produkt, den der Chatbot im Fall von Kaufanfragen bereitstellen kann.
5. **Kategorie**: Eine optionale Spalte, die hilft, Produkte in Kategorien zu unterteilen (z. B. Fahrräder, Zubehör, etc.).

---

### So kannst du die CSV in ein Dataset umwandeln:

```python
from datasets import Dataset
import pandas as pd

# Beispiel CSV-Daten
csv_data = pd.read_csv('produkte.csv')

# Umwandeln in Hugging Face Dataset
dataset = Dataset.from_pandas(csv_data)

# Überprüfen des Datasets
print(dataset)
```