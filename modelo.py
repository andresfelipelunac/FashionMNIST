# Librerias para la creacion de los modelos
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Input, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json

# Otra librerías
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.metrics import confusion_matrix
from itertools import product

def ExplorarDatos():

    print("Obtención y exploración base de datos")

    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    print("Número de imágenes de entrenamiento:",training_images.shape[0])
    print("Número de imágenes de evaluación:",test_images.shape[0])
    print("Tamaño de las imágenes",training_images.shape[1:])

    ### Mostrar ejemplos de imágenes originales
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    nrows = 4
    ncols = 4

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    for i in range(16):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')
        ran = np.random.randint(len(training_labels))
        plt.gca().set_title(class_names[training_labels[ran]])
        plt.imshow(training_images[ran])

    plt.show()

    print("Rango de valores imágenes de entrenamiento: ",np.min(training_images),"-",np.max(training_images))
    print("Rango de valores imágenes de evaluación: ",np.min(test_images),"-",np.max(test_images))

    return training_images,training_labels,test_images, test_labels


def PreProcesarDatos(training_images,training_labels,test_images, test_labels):

    print("Pre-procesamiento de los datos")

    training_images  = training_images / 255.0
    test_images = test_images / 255.0
    training_images.astype('float32')
    test_images.astype('float32')
    training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    print("Nueva dimension imágenes de entrenamiento: ",training_images.shape)
    print("Nueva dimension imágenes de evaluación: ",test_images.shape)
    print("Rango de valores imágenes de entrenamiento: ",np.min(training_images),"-",np.max(training_images))
    print("Rango de valores imágenes de evaluación: ",np.min(test_images),"-",np.max(test_images))

    training_labels = np_utils.to_categorical(training_labels,10)
    test_labels = np_utils.to_categorical(test_labels,10)
    print("Nueva dimensión de los labels de entrenamiento: ",training_labels.shape)
    print("Nueva dimensión de los labels de evaluación: ",test_labels.shape)


def EntrananmientoModelo():

    print("Creación y entrenamiento del modelo")

    modelo = Sequential()
    modelo.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1),padding='same'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.2))
    modelo.add(Conv2D(32, kernel_size=3, activation='relu',padding='same'))
    modelo.add(Dropout(0.2))
    modelo.add(Conv2D(24, kernel_size=3, activation='relu',padding='same'))
    modelo.add(Dropout(0.2))
    modelo.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))
    modelo.add(MaxPooling2D(pool_size=(2,2)))
    modelo.add(Dropout(0.2))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    modelo.summary()

    # Declaración del Callback para detener ejecución
    callback_list=[ModelCheckpoint(filepath='./Modelo/mejorModelo.h5', monitor='val_acc', save_best_only=True, mode='max')]

    modelo.fit(training_images,training_labels,
               validation_data=(test_images,test_labels),
               batch_size=128,
               epochs=20,
               verbose=1,
               callbacks=callback_list)

    # Resultado final
    val_loss,val_acc = modelo.evaluate(test_images,test_labels)
    print("Valor 'loss' de evaluación: ",val_loss)
    print("Valor 'accuracy' de evaluación: ",val_acc)

    # Graficas de la historia del modelos
    accuracy_loss_plots(modelo)

    # Guardar modelo
    GuardarModelo(modelo)


def GuardarModelo(modelo):

    # Guardar arquitectura del modelo
    model_json = modelo.to_json()
    with open("./Model/model_in_json.json", "w") as json_file:
        json.dump(model_json, json_file)

    # Guardar pesos del modelo
    modelo.save_weights("./Model/model_weights.h5")


def CargarModelo():
    with open('./Model/model_in_json.json','r') as f:
        model_json = json.load(f)
    modelo_entrenado = model_from_json(model_json)
    modelo_entrenado.load_weights('./Model/model_weights.h5')
    return modelo_entrenado

def accuracy_loss_plots(model):
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
    ax1.plot(model.history.history['accuracy'], label = 'Accuracy de entrenamiento')
    ax1.plot(model.history.history['val_accuracy'], label = 'Accuracy de evaluación')
    ax1.set_title('Evolución Accuracy por época')
    ax1.set_xlabel('Época')
    ax1.set_ylim(0.8,1)
    ax1.legend()
    ax2.plot(model.history.history['loss'], label='Loss de entrenamiento')
    ax2.plot(model.history.history['val_loss'], label='Loss de evaluación')
    ax2.set_title('Evolución Loss por época')
    ax2.set_xlabel('Época')
    ax2.set_ylim(0,1)
    ax2.legend()

def matrizDeConfusionModelo(modelo):

    # Cargar y preprocesar nuevamente los datos
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images  = training_images / 255.0
    test_images = test_images / 255.0
    training_images.astype('float32')
    test_images.astype('float32')
    training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    training_labels = np_utils.to_categorical(training_labels,10)
    test_labels = np_utils.to_categorical(test_labels,10)

    #Create Multiclass Confusion Matrix
    classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    preds = modelo.predict(test_images)
    cm = confusion_matrix(np.argmax(test_labels,axis=1), np.argmax(preds,axis=1))

    plt.figure(figsize=(8,8))
    plt.imshow(cm,cmap=plt.cm.Reds)
    plt.title('Matriz de Confusión Fashion MNIST')
    plt.colorbar()
    plt.xticks(np.arange(10), classes, rotation=90)
    plt.yticks(np.arange(10), classes)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > 500 else "black")

    plt.tight_layout()
    plt.ylabel('Categoría correcta')
    plt.xlabel('Categoría clasificada')

    plt.show()

    print("Matriz de confusión calculada satisfactoriamente")

def predecirImagenAleatoria(modelo):

    # Cargar y preprocesar nuevamente los datos

    mnist = tf.keras.datasets.fashion_mnist
    classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    _, (test_im, test_lbl) = mnist.load_data()
    indiceImagen = np.random.randint(len(test_lbl))
    test_im2 = test_im.reshape(test_im.shape[0], 28, 28, 1)

    prediccion = modelo.predict(np.expand_dims(test_im2[indiceImagen], axis=0))
    plt.imshow(test_im[indiceImagen])

    print("Clase predecida: ",classes[np.argmax(prediccion)])
    print("Clase correcta: ",classes[test_lbl[indiceImagen]])

    plt.show()

def predecitImagenesPersonalizadas(modelo_entrenado):

    nrows = 3
    ncols = 3
    classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    fig = plt.gcf()
    fig.set_size_inches(ncols*3, nrows*3)

    for i in range (0,9):
        a = cv2.imread("./OtherImages/" + str(i) +".jpg")
        gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        final = cv2.resize(gray, (28,28))
        final = final / 255.0
        prediccion = modelo_entrenado.predict(np.expand_dims(np.expand_dims(final, axis=-1),axis=0))
        sp = plt.subplot(nrows, ncols, i + 1)
        plt.gca().set_title("Clase predecida: " + str(classes[np.argmax(prediccion)]) + "\n Clase real: " + str(classes[i]))
        sp.axis('Off')
        plt.imshow(a)

    plt.show()

if __name__ == '__main__':

    ####### CONSTRUCCION DEL MODELO

    ### Obtención y exploración base de datos
    #training_images,training_labels,test_images, test_labels = ExplorarDatos()

    ### Pre-procesamiento de los datos
    #PreProcesarDatos(training_images,training_labels,test_images, test_labels)

    ### Creación, entrenamiento e historia del modelo
    #EntrananmientoModelo()

    ####### EVALUACIÓN DEL MODELO

    ### Resultados de evaluación del modelo
    #modelo = CargarModelo()
    #matrizDeConfusionModelo(modelo)

    ####### VISUALIACIÓN PREDICCIONES DEL MODELO

    ### Cargar modelo
    #modelo = CargarModelo()

    ### Predecir una imagen de evaluación del Fashion MNIST
    #predecirImagenAleatoria(modelo)

    ### Predecir imágenes en el folder ./Otherimages
    #predecitImagenesPersonalizadas(modelo)
