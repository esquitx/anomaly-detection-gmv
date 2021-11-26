import parametros
from autoencoder import losses
from autoencoder import metrics

import os
import datetime

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Postprocessing
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte

import json

# AUXILIARES
def cargar_modelo(path):
    
    carpeta = path
    
    # INFO
    with open(os.path.join(carpeta, "info.json"), "r") as read_file:
        info = json.load(read_file)

    loss = info["modelo"]["loss"]
    rango = info["preprocesamiento"]["rango"]
    modelo = info["modelo"]["tipo"]

    dir_modelo = os.path.join(carpeta, "{}.hdf5".format(modelo))
    if loss == "mssim":
        modelo = keras.models.load_model(
            filepath=dir_modelo,
            custom_objects = {
                "LeakyReLU": keras.layers.LeakyReLU,
                "loss" : losses.mssim(rango), 
                "ssim" : metrics.metrics(rango),
            },
            compile=False,
        )
        modelo.compile(loss=losses.mssim(rango), metrics=metrics.metrics(rango))
    elif loss == "l2":
        modelo = keras.models.load_model(
            filepath=dir_modelo,
            custom_objects = {
                "LeakyReLU": keras.layers.LeakyReLU,
                "l2_loss" : losses.l2_loss,
                "ssim" : losses.ssim(rango), 
                "mssim" : metrics.metrics(rango),
            },
            compile=False,
        )
        modelo.compile(loss=losses.l2_loss, metrics=metrics.metrics(rango))

    return modelo, info

# PREPROCESSING
class GeneradorDatos:
    def __init__(
        self, dir_datos, rescale, shape):

        self.dir_datos = dir_datos
        self.dir_train = os.path.join(dir_datos, "train")
        self.dir_test = os.path.join(dir_datos, "test")
        self.rescale = rescale
        self.shape = shape
    
        self.validation_split = parametros.VAL_SPLIT

    
    # Utilizamos ImageDataGenerator : https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # Genera variaciones de imágenes dados unos parámetros
    # Es útil para que el modelo sea capaz de reconocer errores independientemente de la posicion de los pxs.
    

    def datos_train(self, batch_size, shuffle=True):
        # generamos datos extra
        generacion_train = ImageDataGenerator(
            
            # rotamos imágenes aleatoriamente
            rotation_range=parametros.ANGULO,
            
            # Desplazamos imagen horizontalmente
            width_shift_range=parametros.DESPLAZAMIENTO_HORIZONTAL,
            # Desplazamos imagen verticalmente
            height_shift_range=parametros.DESPLAZAMIENTO_VERTICAL,

            # En el caso de que la imagen quede más pequeña que 
            # la forma deseada, método que se usa para llenar los vacíos
            fill_mode=parametros.LLENADO,

            # Ajuste de brillo. Es un rango en torno al 1.
            brightness_range=parametros.BRILLO,

            # Primera transformación. Ajuste de tamaño dentro del canvas
            rescale=self.rescale,
            
            # Establece la forma de la imagen a [FILAS, COLUMNAS, CANALES]
            data_format="channels_last",

            # Imagenes de validacion
            validation_split=self.validation_split,
        )

        # Obtiene imagenes de train y crea un dataset con las transformaciones anteriores
        dataset_train = generacion_train.flow_from_directory(
            directory=self.dir_train,
            target_size=self.shape,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="input",
            subset="training",
            shuffle=True,
        )
        return dataset_train

    def datos_val(self, batch_size, shuffle=True):
        # Como solo queremos validar las imágenes, no hace falta transformarlas
        generacion_val = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            validation_split=self.validation_split,
        )
        # Generate validation batches with datagen.flow_from_directory()
        dataset_val = generacion_val.flow_from_directory(
            directory=self.dir_train,
            target_size=self.shape,
            batch_size=batch_size,
            class_mode="input",
            subset="validation",
            shuffle=shuffle,
        )
        return dataset_val

    def datos_test(self, batch_size, shuffle=False):
        
        # Igual que en validación, solo ajustamos el tamaño
        generacion_test = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
        )

        dataset_test = generacion_test.flow_from_directory(
            directory=self.dir_test,
            target_size=self.shape,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return dataset_test

    def num_imagenes_test(self):
        n = 0
        subdirs = os.listdir(self.dir_test)
        for subdir in subdirs:
            path_subdir = os.path.join(self.dir_test, subdir)
            imagenes = os.listdir(path_subdir)
            n += len(imagenes)
        return n


# GUARDAR EL MODELO
def guardar(modelo, datos_train, datos_val,nombre_modelo, dir_datos, loss, epochs, batch_size):


        tiempo = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        dir_guardado = os.path.join(
            os.getcwd(),
            "modelos_guardados",
            "normal", 
            parametros.LOSS,
            tiempo,
        )

        if not os.path.isdir(dir_guardado):
            os.makedirs(dir_guardado)
        

        modelo.save(os.path.join(dir_guardado, "{}.hdf5".format(nombre_modelo)))

        info = {
            "datos" : {
            "dir_datos" :  dir_datos,
            "num_imagenes_train" : datos_train.samples,
            "num_imagenes_validacion" : datos_val.samples,
            "val_split" : parametros.VAL_SPLIT
            },

            "modelo" : {"tipo" : "normal", "loss" : loss},

            "preprocesamiento" : {
                "color" : "rgb",
                "rescale" : modelo.RESCALE,
                "shape" : modelo.SHAPE,
                "val_min" : modelo.val_min,
                "val_max" : modelo.val_max, 
                "rango" : modelo.rango, 

            },

            "training" : {
                "batch_size" : batch_size, 
                "epochs" : epochs, 
            },
        }

        with open(os.path.join(dir_guardado, "info.json"), "w") as archivo_info:
            json.dump(info, archivo_info, indent=4)
        

        print("Archivos guardados en \n{}\n".format(dir_guardado))

        return None

# POSTPROCESAMIENTO ( CLASIFICACIÓN)
class ArrayImagen:
    def __init__(self, entrada, predicciones, val_min, val_max, metodo, dtype="float64", nombre_imagenes=None,):
        
        # val_min y val_max son los valores min/max de los píxeles de las ímagenes
        # tanto de entrada y salida
        self.val_min = val_min
        self.val_max = val_max
        self.nombre_imagenes = nombre_imagenes

        # Error
        self.metodo = metodo

        # Formato para los valores de los píxeles ()
        self.dtype = dtype


        self.entrada = entrada
        self.predicciones = predicciones

        # Cálculamos el mapa de resolución
        self.scores, self.resmaps = calcular_resmap(self.entrada, self.predicciones, self.metodo)

        self.val_min_resmap = 0.0
        self.val_max_resmap = 1.1

# RESMAPS
def calcular_resmap(entrada, prediccion, metodo="ssim", dtype="float64"):
    
    # Transformamos a escala de grises y a un tensor dimensión 3
    imagenes_gris = tf.image.rgb_to_grayscale(entrada).numpy()[:, :, :, 0]
    predicciones_gris = tf.image.rgb_to_grayscale(prediccion).numpy()[:, :, :, 0]

    if metodo=="l2":
        puntuaciones, resmaps = resmaps_l2(imagenes_gris, predicciones_gris)
    elif metodo=="ssim":
        puntuaciones, resmaps = resmaps_ssim(imagenes_gris, predicciones_gris)

    return puntuaciones, resmaps

def resmaps_l2(entrada, predicciones):
    resmaps = (entrada - predicciones) ** 2
    puntuaciones = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return puntuaciones, resmaps

def resmaps_ssim(entrada, predicciones):
    resmaps = np.zeros(shape=entrada.shape, dtype="float64")
    puntuaciones = []

    for i in range(len(entrada)):

        img_entrada, img_pred = entrada[i], predicciones[i]
        

        puntuacion, resmap = structural_similarity(img_entrada, img_pred, win_size=11, gaussian_weights=True, multichannel=False, sigma=1.5, full=True,)

        resmaps[i] = 1 - resmap
        puntuaciones.append(puntuacion)


    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return puntuaciones, resmaps


def clasificar_imagenes(imagenes):

    imagenes_clasificadas = np.zeros(shape=imagenes.shape)
    total_areas = []

    for i, imagen in enumerate(imagenes):
        
        # Quitamos los bordes
        imagen_preparada = clear_border(imagen)

        # Clasificamos la imagen y guardamos su tipo

        imagen_clasificada = label(imagen_preparada)
        imagenes_clasificadas[i] = imagen_clasificada
        regiones = regionprops(imagen_clasificada)
        
        if regiones: 
            areas = [region.area for region in regiones]
            total_areas.append(areas)
        else:
            total_areas.append([0])
    
    return imagenes_clasificadas, total_areas

        
