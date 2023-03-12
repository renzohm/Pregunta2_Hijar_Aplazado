from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

# Lee las imágenes y etiquetas de la carpeta "shapes"
folder = './train/cuadrados'
images = []
labels = []
for filename in os.listdir(folder):
    if filename.endswith('.png'):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((64, 64)) # Cambia el tamaño de la imagen a 64 x 64
        img_array = np.array(img)
        images.append(img_array)
        labels.append([1,0,0])  # La clase se encuentra en el nombre del archivo

for filename in os.listdir('./train/circulos'):
    if filename.endswith('.png'):
        img = Image.open(os.path.join('./train/circulos', filename))
        img = img.resize((64, 64)) # Cambia el tamaño de la imagen a 64 x 64
        img_array = np.array(img)
        images.append(img_array)
        labels.append([0,1,0])  # La clase se encuentra en el nombre del archivo

for filename in os.listdir('./train/triangulos'):
    if filename.endswith('.png'):
        img = Image.open(os.path.join('./train/triangulos', filename))
        img = img.resize((64, 64)) # Cambia el tamaño de la imagen a 64 x 64
        img_array = np.array(img)
        images.append(img_array)
        labels.append([0,0,1])  # La clase se encuentra en el nombre del archivo

# Codifica las etiquetas como números
encoder = LabelEncoder()
# labels = encoder.fit_transform(labels)

# # Divide el conjunto de datos en entrenamiento y prueba
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.4)

# # Convierte los conjuntos de datos a arreglos NumPy
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# Define la arquitectura de la red neuronal
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compila el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))

# Evalúa el modelo con el conjunto de datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

def clasificar(value):
    if(value==0):
        return 'Cuadrado'
    elif(value==1):
        return 'Círculo'
    return 'Triángulo'


# Carga una imagen de prueba
img = np.array(Image.open('./prueba/circulo.jpg').resize((64, 64)))

# # Normaliza la imagen
img = img / 255.0

# # Agrega una dimensión para convertirla en un lote de una imagen
img = np.expand_dims(img, axis=0)

# # Obtiene la predicción de la red neuronal
pred = model.predict(img)

print("La imagen es un " + clasificar(pred))