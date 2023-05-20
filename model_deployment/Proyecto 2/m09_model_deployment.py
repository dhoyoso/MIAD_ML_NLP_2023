#!/usr/bin/python

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import joblib
import spacy
import nltk
import re
import os

# Cargar modelo de spacy
model_loaded = spacy.load("./models/")

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Crear objeto para lematización
lemmatizer = WordNetLemmatizer()

# Obtener stopwords en inglés
stopwords = set(stopwords.words('english'))

# Preprocesar los textos
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords)

    # Lematización
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    
    # Eliminar caracteres especiales
    text = re.sub(r'[^a-zA-Z ]+', '', text)

    return text

# Pre procesar entrada
def pre_process(entry):

    # Transformación 1 --> Juntar title con plot en plot.
    entry['plot'] = entry['title'] + ', ' + entry['plot']

    # Transformación 2 --> Convertir a minusculas, eliminar stopwords, lematizar y eliminar caracteres especiales.
    entry['plot'] = entry['plot'].apply(preprocess_text)

    return entry['plot'][0]

def predict(year, title, plot, rating):

    # Cargar entrada
    entry = pd.DataFrame([[year, title, plot, rating]], columns=['year','title','plot','rating'])
    entry = entry.astype({'year':int, 'title':str, 'plot':str, 'rating':float})
  
    # Pre process entrada
    text = pre_process(entry)

    # Realizar prediccion
    doc = model_loaded(text)
    classes = [label for label in doc.cats if doc.cats[label] >= 0.5]

    # Poner el prefijo P_ que denota probabilidad
    temp = "P_"
    predictions = {temp + str(key): val for key, val in doc.cats.items()}

    # Poner las clases con mayor probabilidad (mayores a 0.5) en una llave aparte del diccionario final
    predictions['PRED_CLASSES'] = ', '.join(classes)

    return predictions