#!/usr/bin/python

from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import keras
import nltk
import re
import os


# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')


# Set CUDA_VISIBLE_DEVICES to an empty string
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Columnas de la predicción
cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

# Recrea exactamente el mismo modelo desde el archivo
model = keras.models.load_model('final_model.h5')

# Recrea el tokenizador
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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
    max_length = 300
    padding_type='post'
    truncation_type='post'

    # Transformación 1 --> Juntar title con plot en plot.
    entry['plot'] = entry['title'] + '; ' + entry['plot']

    # Transformación 2 --> Convertir a minusculas, eliminar stopwords, lematizar y eliminar caracteres especiales.
    entry['plot'] = entry['plot'].apply(preprocess_text)

    # Transformación 3 --> Tokenizar texto
    X_test_seq = tokenizer.texts_to_sequences(entry['plot'])

    # Transformación 4 --> Realizar padding al dataset
    X_test_pad = pad_sequences(X_test_seq,maxlen=max_length, 
                                padding=padding_type, truncating=truncation_type)

    # Retorna la entrada pre procesada
    return X_test_pad

def predict(year, title, plot):

    # Cargar entrada
    entry = pd.DataFrame([[year, title, plot]], columns=['year','title','plot'])
    entry = entry.astype({'year':int, 'title':str, 'plot':str})
  
    # Pre process entrada
    processed_entry = pre_process(entry)

    # Realizar prediccion
    preds = model.predict(processed_entry)
    print(preds)
    # Post procesar salida, poner P_género a cada probabilidad y convertir a diccionario
    preds_df = pd.DataFrame(preds, columns=cols)
    pred_dict = preds_df.iloc[0].to_dict()

    # Poner las clases con mayor probabilidad (mayores a 0.5) en una llave aparte del diccionario final
    pred_classes = []
    for k,v in pred_dict.items():
        if v >= 0.5:
            pred_classes.append(k.replace('p_',''))
            
    pred_dict['PRED_CLASSES'] = ', '.join(pred_classes)

    return pred_dict