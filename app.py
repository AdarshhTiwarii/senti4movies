from flask import Flask, render_template, url_for, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)
Swagger(app)

mnb = pickle.load(open('Naive_Bayes_model_imdb.pkl', 'rb'))
countVect = pickle.load(open('countVect_imdb.pkl', 'rb'))

tfidfVect = pickle.load(open('tfidf_imdb.pkl', 'rb'))
model = joblib.load('tfidfLR.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]

        my_prediction = model.predict(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
