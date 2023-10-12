#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, r2_score, f1_score

#Carga de archivos
titanic = sns.load_dataset(name = "titanic")
print(titanic.head(2))

#Rellenamos nulos
titanic_limpio = titanic.ffill().bfill()

#Declaramos variables dependientes e independientes para la regresión logística
X = titanic_limpio[["pclass", "age", "fare"]]
y = titanic_limpio["alive"]

print(titanic_limpio["pclass"].value_counts())
print(titanic_limpio["age"].value_counts())
print(titanic_limpio["fare"].value_counts())

#Hacemos validación cruzada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45, random_state = None)

#Se escalan todos los datos
escalar = StandardScaler()

#Para realizar el escalamiento de las variables "X" tanto de entrenamiento como de prueba, utilizando
#Y no la normalizamos porque es categórica
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)

#Definimos el algoritmo a utilizar
algoritmo = LogisticRegression()

#Entrenamos el modelo
algoritmo.fit(X_train, y_train)
print(algoritmo.fit)

#Realizamos la predicción
y_pred = algoritmo.predict(X_test)
print(y_pred)

#Verificamos la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión")
print(matriz)

#Calculo de la precisión del modelo
precision = precision_score(y_test, y_pred, average = "binary", pos_label = "no")
print("Precisión: ", precision)

#Calculo de la exactitud del modelo
exactitud = accuracy_score(y_test, y_pred)
print("Exactitud: ", exactitud)

#Calculo de la sensibilidad del modelo
sensibilidad = recall_score(y_test, y_pred, average = "binary", pos_label = "yes") 
print("Sensibilidad:", sensibilidad)

#Calculamos Puntaje F1
puntajef1 = f1_score(y_test, y_pred, average = "binary", pos_label = "yes")
print("Puntaje F1: ", puntajef1)

