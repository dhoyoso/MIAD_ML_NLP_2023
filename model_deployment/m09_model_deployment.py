#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict(year, mileage, state, make, model):

    modelo = joblib.load(os.path.dirname(__file__) +'/model.pkl') 
    encoder = joblib.load(os.path.dirname(__file__) +'/encoder.pkl') 

    entry = pd.DataFrame([[year, mileage, state, make, model]], columns=['Year','Mileage','State','Make','Model'])
    
    entry = entry.astype({ 'Year':int, 'Mileage':int, 'State':str, 'Make':str, 'Model':str})
  
    # Pre process features
    entry = encoder.transform(entry)

    # Make prediction
    p1 = modelo.predict(entry)
    
    return p1[0]