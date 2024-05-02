from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
# flask app
app = Flask(__name__)

log_reg_model = pickle.load(open('models/Logistic.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))


def inputEnglish(new_input_text):
    new_input_tfidf = tfidf_vectorizer.transform([new_input_text])
    # Predi ct the sentiment using the trained Logistic Regression model
    predicted_sentiment = log_reg_model.predict(new_input_tfidf)[0]
    if predicted_sentiment == 0:
        return "Neutral"
    elif predicted_sentiment == 1:
        return ("Positive")
    else:
        return ("Negative")


# routes


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page


@app.route('/convert', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form.get('valEnglish')

        if input_text == '':
            message = "Please Fill before submit."
            return render_template('index.html', message=message)
        else:
            predicted = inputEnglish(input_text)
            print(predicted)
            return render_template('index.html', predicted=predicted)

    return render_template('index.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
