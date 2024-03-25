import dill
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from joblib import load
from dill import load
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings, string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

with open('model.dill', 'rb') as f:
    pipeline = dill.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predictions = pipeline.predict([data['text']])

    return jsonify({
        'is_fake_review': predictions.tolist()[0] == 'CG'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
