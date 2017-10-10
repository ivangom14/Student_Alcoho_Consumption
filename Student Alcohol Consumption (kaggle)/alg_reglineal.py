import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.pipeline import Pipeline
import transformers
from sklearn import svm
import sys
from sklearn.externals import joblib
import getopt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split

datos = pd.read_csv("student-por.csv", sep=";")
print("Tamanio dataset inicial", datos.shape)


pipeline = Pipeline([
    ('Binary_school', transformers.Encoder('school')),
    ('Binary_sex', transformers.Encoder('sex')),
    ('Binary_address', transformers.Encoder('address')),
    ('Binary_famsize', transformers.Encoder('famsize')),
    ('Binary_Pstatus', transformers.Encoder('Pstatus')),
    ('Enconder_Mjob', transformers.Encoder('Mjob')),
    ('Enconder_Fjob', transformers.Encoder('Fjob')),
    ('Enconder_reason', transformers.Encoder('reason')),
    ('Enconder_guardian', transformers.Encoder('guardian')),
    ('Binary_schoolsup', transformers.Encoder('schoolsup')),
    ('Binary_famsup', transformers.Encoder('famsup')),
    ('Binary_paid', transformers.Encoder('paid')),
    ('Binary_activities', transformers.Encoder('activities')),
    ('Binary_nursery', transformers.Encoder('nursery')),
    ('Binary_higher', transformers.Encoder('higher')),
    ('Binary_internet', transformers.Encoder('internet')),
    ('Binary_romantic', transformers.Encoder('romantic')),
#    ('Total_time_non_school',transformers.Adder('traveltime','studytime','totaltime')),
#    ('Add_decomposition_dataset',transformers.Decomposition(n_components=4)),
#    ('Borrar', transformers.Columns_dropper(['Medu','Fedu'])),
#    ('Normalize_dataset', transformers.Normalizer())

])

dataset = pipeline.transform(datos)

dataset= dataset.set_index(["G3"])
#dataset = dataset.drop(["G1","G2"],axis=1)
indice = dataset.index
datos = dataset.values


datos_train, datos_test, indice_train, indice_test = train_test_split(datos, indice)

print("Tamanio indice test", indice_test.shape)
print("Tamanio datos test", datos_test.shape)

print("Tamanio indice train", indice_train.shape)
print("Tamanio datos train", datos_train.shape)

# Entrenamiento del modelo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

regresion_lineal = LinearRegression()
regresion_lineal.fit(datos_train, indice_train)

coef = regresion_lineal.coef_ 
#print("Coeficientes regresión lineal", coef)
term = regresion_lineal.intercept_ 
#print("Termino independiente regresion lineal", term)

#res = regresion_lineal.residues_ #  Suma de los residuos
#print("Residuos regresión lineal", res)

print("G3 test", indice_test)
indice_pred = regresion_lineal.predict(datos_test) 
indice_pred = indice_pred.round(decimals=0)
print("Predicción de G3",indice_pred)
errormedio = mean_absolute_error(indice_test, indice_pred) 
print("Error medio de predicción", errormedio)


