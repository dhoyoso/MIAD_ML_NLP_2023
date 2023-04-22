#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict(year, mileage, state, make, model):

    modelo = joblib.load('C:/Users/dhoyoso/Documents/Maestria UNIANDES/Cursos/MACHINE LEARNING Y PROCESAMIENTO LENGUAJE NATURAL/MIAD_ML_NLP_2023/model_deployment/model.pkl') 
    encoder = joblib.load('C:/Users/dhoyoso/Documents/Maestria UNIANDES/Cursos/MACHINE LEARNING Y PROCESAMIENTO LENGUAJE NATURAL/MIAD_ML_NLP_2023/model_deployment/encoder.pkl') 

    entry = pd.DataFrame([[year, mileage, state, make, model]], columns=['Year','Mileage','State','Make','Model'])
    
    entry = entry.astype({ 'Year':int, 'Mileage':int, 'State':str, 'Make':str, 'Model':str})
  
    # Pre process features
    entry = encoder.transform(entry)

    # Make prediction
    p1 = modelo.predict(entry)
    
    return p1[0]