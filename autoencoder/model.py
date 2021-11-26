# Recursos: 
# https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/

# Inspired by AndneneBounessouer/MVTec-Anomaly-Detection

import os
import datetime
import json
from pathlib import Path
import six
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

from tensorflow.keras.layers import(
    BatchNormalization,
    GlobalAveragePooling2D,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    LeakyReLU, 
    Activation,
    Reshape,
    Flatten, 
    Dense,
    Input, 
    Add, 
    ReLU,
    AveragePooling2D,
    concatenate
)

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K

from autoencoder import (
    metrics, 
    losses
)

import parametros

def lr_scheduler(epoch, lr):
    decay = parametros.DECAY
    step = parametros.STEP

    if epoch % step == 0:
        return lr * decay
    
    return lr

class AutoencoderNormal():
    def __init__(self):

        # PARÁMETROS
        self.rescale = parametros.info_resnet["rescale"]
        self.shape = parametros.info_resnet["shape"]
        self.val_min = parametros.info_resnet["val_min"]
        self.val_max = parametros.info_resnet["val_max"]
        self.rango = parametros.info_resnet["rango"]


    @staticmethod
    def crear_modelo(shape):

        dimension = (*shape, 3)
        dimension_encoding = 64    

        imagen_entrada = Input(shape=dimension)
        # ENCODER
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(imagen_entrada)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        

        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        

        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        

        x = Flatten()(x)
        x = Dense(dimension_encoding, kernel_regularizer=l2(1e-6))(x)
        x = LeakyReLU(alpha=0.1)(x)
        

        # DECODED
        x = Reshape((4, 4, dimension_encoding // 16))(x)
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        
        x = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)


        x = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        
        x = Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)
        # ---------------------------------------------------------------------------------

        x = Conv2D(3, (3, 3), padding="same", kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Activation("sigmoid")(x)
        
        decoded = x

        autoencoder = keras.Model(imagen_entrada, decoded)

        return autoencoder



""" 
RESNET BLOCKS
---------------------
BN -> RELU : BatchNormalization to ReLU
CONV -> BN -> RELU : Convolutional Filter to BatchNormalization to ReLU
BN -> RELU -> CONV : Reverse previous filter. BatchNormalization to ReLU to convolutional filter
SHORTCUT BLOCK :  residual function block
BOTTLENECK BLOCK : Adjusts filtros if network has > 34 layers
RESIDUAL BLOCK : Block built by bottleneck blocks
BASIC BLOCK : 3x3 convolutional filter block for networks with < 34 layers

 """

global ROW_AXIS
global COLUMN_AXIS
global CHANNEL_AXIS

ROW_AXIS = 1
COLUMN_AXIS = 2
CHANNEL_AXIS = 3


def bn_a_relu(entrada):
    # Usamos 3 de axis porque utilizamos RGB
    normalizado = BatchNormalization(axis=CHANNEL_AXIS)(entrada)
    return Activation("relu")(normalizado)


def conv_a_bn_a_relu(**parametros_convolucional):
    """Helper to build a conv -> BN -> relu block
    """
    filtros = parametros_convolucional["filtros"]
    kernel_size = parametros_convolucional["kernel_size"]
    strides = parametros_convolucional.setdefault("strides", (1, 1))
    kernel_initializer = parametros_convolucional.setdefault("kernel_initializer", "he_normal")
    padding = parametros_convolucional.setdefault("padding", "same")
    kernel_regularizer = parametros_convolucional.setdefault("kernel_regularizer", l2(1.0e-4))

    def f(entrada):
        conv = Conv2D(
            filters=filtros,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )(entrada)
        return bn_a_relu(conv)

    return f


def bn_a_relu_a_conv(**parametros_convolucional):
    """ BLOQUE BathNormalization -> ReLU -> Conv"""
    filtros = parametros_convolucional["filtros"]
    kernel_size = parametros_convolucional["kernel_size"]
    strides = parametros_convolucional.setdefault("strides", (1, 1))
    kernel_initializer = parametros_convolucional.setdefault("kernel_initializer", "he_normal")
    padding = parametros_convolucional.setdefault("padding", "same")
    kernel_regularizer = parametros_convolucional.setdefault("kernel_regularizer", l2(1.0e-4))

    def f(entrada):
        activacion= bn_a_relu(entrada)
        return Conv2D(
            filters=filtros,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )(activacion)

    return f


def atajo(entrada, residual, ultima_capa=False):
    """Ataja el flujo del model conectando el layer de entrada con el bloque residual 
    """

    entrada_shape = K.int_shape(entrada)
    residual_shape = K.int_shape(residual)
    stride_ancho = int(round(entrada_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_alto = int(round(entrada_shape[COLUMN_AXIS] / residual_shape[COLUMN_AXIS]))
    mismos_canales = entrada_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    atajo = entrada
    # Aplicamos identidad si la forma es igual, si no 1x1
    if stride_ancho > 1 or stride_alto > 1 or not mismos_canales:
        atajo = Conv2D(
            filters=residual_shape[CHANNEL_AXIS],
            kernel_size=(1, 1),
            strides=(stride_ancho, stride_alto),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1.0e-4),
        )(entrada)

    if ultima_capa:
        return atajo

    return concatenate([atajo, residual])


def bloque_residual(tipo_bloque, filtros, repeticiones, primera_capa=False, ultima_capa=False):
    def f(entrada):
        for i in range(repeticiones):
            strides_iniciales = (1, 1)

            if i == 0 and not primera_capa:
                strides_iniciales = (2, 2)

            entrada = tipo_bloque(
                filtros=filtros,
                strides_iniciales=strides_iniciales,
                primer_bloque_primera_capa=(primera_capa and i == 0),
                ultima_capa=ultima_capa)(entrada)

        return entrada

    return f


def bloque(filtros, strides_iniciales=(1, 1), primer_bloque_primera_capa=False, ultima_capa=False,):

    """ Bloque convolucional 3x3 para redes con < 34 capas según : http://arxiv.org/pdf/1603.05027v2.pdf"""

    def f(entrada):

        if primer_bloque_primera_capa:
            # Si es primer bloque, no hace falta BN -> ReLU
            conv1 = Conv2D(
                filters=filtros,
                kernel_size=(3, 3),
                strides=strides_iniciales,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4),
            )(entrada)
        else:
            conv1 = bn_a_relu_a_conv(
                filtros=filtros, kernel_size=(3, 3), strides=strides_iniciales
            )(entrada)

        residual = bn_a_relu_a_conv(filtros=filtros, kernel_size=(3, 3))(conv1)
        return atajo(entrada, residual, ultima_capa)

    return f


def bottleneck(filtros, strides_iniciales=(1, 1), primer_bloque_primera_capa=False, ultima_capa=False):
    """
    Bloque bottleneck en el caso de que la red > 34 capas :  http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(entrada):

        if primer_bloque_primera_capa:
            conv_1_1 = Conv2D(
                filters=filtros,
                kernel_size=(1, 1),
                strides=strides_iniciales,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4),)(entrada)
        else:
            conv_1_1 = bn_a_relu_a_conv(
                filtros=filtros, kernel_size=(1, 1), strides=strides_iniciales)(entrada)

        conv_3_3 = bn_a_relu_a_conv(filtros=filtros, kernel_size=(3, 3))(conv_1_1)
        residual = bn_a_relu_a_conv(filtros=filtros * 4, kernel_size=(1, 1))(conv_3_3)

        return atajo(entrada, residual)

    return f


bloques = {
    "bloque": bloque, 
    "bottleneck": bottleneck
}
def que_bloque(nombre):
    if type(nombre) == str:
        return bloques[nombre]
    else: 
        return nombre

class ResnetBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def build(dim_entrada, tipo_bloque, repeticiones):
        """Construye el modelo
        Args:
            entrada: tamaño entrada con canales primero (canales, filas, columnas)
            salida: número de elementos de salida
            tipo_bloque: Tipo de blockes a utilizar.
                The original paper used bloque for layers < 50
            cuantos blcoques de cada tipo: Number of repetitions of various block units.
                En cada repetición del bloque, el número de filtros se duplica y la entrada se reduce a la mitad

        Returns:
            Modelo
        """


        tipo_bloque = que_bloque(tipo_bloque)

        entrada = Input(shape=dim_entrada)
        conv1 = conv_a_bn_a_relu(filtros=64, kernel_size=(7, 7), strides=(2, 2))(entrada)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        bloque = pool1
        filtros = 64
        for i, r in enumerate(repeticiones):
            bloque = bloque_residual(
                tipo_bloque,
                filtros=filtros,
                repeticiones=r,
                primera_capa=(i == 0),
                ultima_capa=(i == 3),)(bloque)
            filtros *= 2


        model = Model(inputs=entrada, outputs=bloque)
        return model

    # Diferentes tipos
    @staticmethod
    def build_resnet_18(dim_entrada):
        return ResnetBuilder.build(dim_entrada, bloque, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(dim_entrada):
        return ResnetBuilder.build(dim_entrada, bloque, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(dim_entrada):
        return ResnetBuilder.build(dim_entrada, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(dim_entrada,):
        return ResnetBuilder.build(dim_entrada, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(dim_entrada):
        return ResnetBuilder.build(dim_entrada, bottleneck, [3, 8, 36, 3])


class AutoencoderResidual:
    
    def __init__(self):
        pass

    @staticmethod
    def crear_modelo(shape):
        channels = 3
        resnet = ResnetBuilder.build_resnet_152((*shape, channels))
        
        x = Conv2D(512, (1, 1), strides=1, activation="relu", padding="valid")(resnet.output)

        encoded = Conv2D(512, (1, 1), strides=1, activation="relu", padding="valid")(x)

        # decoder
        layer_1 = Conv2DTranspose(
            512, kernel_size=4, strides=2, padding="same", activation=None,
        )(encoded)
        layer_2 = BatchNormalization()(layer_1)
        layer_3 = ReLU()(layer_2)
        layer_4 = Conv2DTranspose(
            512, kernel_size=3, strides=1, padding="SAME", activation=None,
        )(layer_3)
        layer_5 = BatchNormalization()(layer_4)
        layer_6 = ReLU()(layer_5)
        ####
        add_1 = Add()([layer_1, layer_6])
        ####
        layer_7 = Conv2DTranspose(
            256, kernel_size=4, strides=2, padding="same", activation=None,
        )(add_1)
        layer_8 = BatchNormalization()(layer_7)
        layer_9 = ReLU()(layer_8)
        layer_10 = Conv2DTranspose(
            256, kernel_size=3, strides=1, padding="SAME", activation=None,
        )(layer_9)
        layer_11 = BatchNormalization()(layer_10)
        layer_12 = ReLU()(layer_11)
        ####
        add_2 = Add()([layer_7, layer_12])
        ####
        layer_13 = Conv2DTranspose(
            128, kernel_size=4, strides=2, padding="SAME", activation=None,
        )(add_2)
        layer_14 = BatchNormalization()(layer_13)
        layer_15 = ReLU()(layer_14)
        layer_16 = Conv2DTranspose(
            128, kernel_size=3, strides=1, padding="SAME", activation=None,
        )(layer_15)
        layer_17 = BatchNormalization()(layer_16)
        layer_18 = ReLU()(layer_17)
        ####
        add_3 = Add()([layer_13, layer_18])
        ####
        layer_19 = Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", activation=None,
        )(add_3)
        layer_20 = BatchNormalization()(layer_19)
        layer_21 = ReLU()(layer_20)
        layer_22 = Conv2DTranspose(
            64, kernel_size=3, strides=1, padding="SAME", activation=None,
        )(layer_21)
        layer_23 = BatchNormalization()(layer_22)
        layer_24 = ReLU()(layer_23)
        ####
        add_4 = Add()([layer_19, layer_24])
        ####
        decoded = Conv2DTranspose(channels, kernel_size=4, strides=2, padding="same", activation="sigmoid",)(add_4)

        model = Model(resnet.input, decoded)

        return model
