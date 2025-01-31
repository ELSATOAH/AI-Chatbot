#import cohere
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a secret key for CSRF protection

# Vordefinierte Antworten des Chatbots
antworten = {   
    "hallo": "Hallo! Wie kann ich dir helfen?", 
    "wie geht es dir?": "Mir geht es gut, danke der Nachfrage!", 
    "was kannst du?": "Ich kann einfache Fragen beantworten.", 
    "tschüss": "Auf Wiedersehen!",
}

class Form(FlaskForm):
    text = StringField('Enter text to search', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = Form()
    antwort = None  # Standardwert setzen

    if request.method == 'POST':
        text = request.form['text'].lower()
        antwort = antworten.get(text, "Entschuldigung, ich habe dich nicht verstanden.")

    return render_template('Homepage.html', form=form, output=antwort)



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
