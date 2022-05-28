#Importacion de Librerias
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#Datos para la red neuronal
datos={
"Primero":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
"Segundo":[1,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
}

#Visualizacion de los datos
df= pd.DataFrame(datos)
print(df)
plt.scatter(datos["Primero"],datos["Segundo"])
plt.show()

#Variables para el entrenamiento
Ytrain= datos["Segundo"]
Xtrain= datos["Primero"]

#Creacion de el modelo
modelo= keras.Sequential()
modelo.add(keras.layers.Dense(units=1, input_shape=[1]))

#Compilacion de el modelo
modelo.compile(
    optimizer= keras.optimizers.Adam(1),
    loss= 'mean_squared_error'
)

#Entrenamiento de el modelo
epochs_hist= modelo.fit(Xtrain, Ytrain, epochs= 100)

#Visualizacion de el entrenamiento
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de Perdida Durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Entrenamineto Perdido')
plt.legend("Entrenamiento Perdido")
plt.show()

#Hacer Predicciones
Primero= int(input("Ingrese Numero: "))
Temp_F= modelo.predict([Primero])
print('Prediccion de la Red Neuronal: ' + str(Temp_F))