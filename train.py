from autoencoder.model import (
    AutoencoderNormal, 
    lr_scheduler, 
    AutoencoderResidual
)

from processing import (
    GeneradorDatos,
    guardar
)

import parametros
from autoencoder import losses
from autoencoder import metrics

import os
import argparse
import datetime
import json
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def guardar(dir_guardado, dir_datos, datos_train, datos_val, modelo, tipo, batch_size, epochs, loss):

    
    modelo.save(os.path.join(dir_guardado, "{}.hdf5".format(tipo)))

    if tipo == "normal":
        datos = parametros.info_normal
    elif tipo == "resnet":
        datos = parametros.info_resnet

    info = {
        "datos" : {
        "dir_datos" :  dir_datos,
        "num_imagenes_train" : int(epochs * datos_train.samples),
        "num_imagenes_validacion" : int(epochs * datos_val.samples),
        "val_split" : parametros.VAL_SPLIT
        },

        "modelo" : {"tipo" : "normal", "loss" : loss},

        "preprocesamiento" : {
            "color" : "rgb",
            "rescale" : datos["rescale"],
            "shape" : datos["shape"],
            "val_min" : datos["val_min"],
            "val_max" : datos["val_max"], 
            "rango" : datos["rango"], 
        },

        "training" : {
            "batch_size" : batch_size, 
            "epochs" : epochs, 
            "num_imagenes_train" : int(epochs* datos_train.samples),
        },
    }

    with open(os.path.join(dir_guardado, "info.json"), "w") as archivo_info:
        json.dump(info, archivo_info, indent=4)
    

    print("Archivos guardados en \n{}\n".format(dir_guardado))

    return None

def main(args):

    # Obtenemos datos para el entrenamiento
    dir_datos = args.dir_datos
    tipo = args.tipo
    loss = args.loss
    lr = parametros.INIT_LR
    epochs = parametros.EPOCHS
    batch_size = parametros.BATCH_SIZE

    if tipo == "normal":
        info = parametros.info_normal
        autoencoder = AutoencoderNormal.crear_modelo(parametros.info_normal["shape"])
    elif tipo == "resnet":
        info = parametros.info_resnet
        autoencoder = AutoencoderResidual.crear_modelo(parametros.info_resnet["shape"])
    
    print("Construyendo datos de entrenamiento...")
    preprocesado = GeneradorDatos(
        dir_datos=dir_datos,
        rescale=info["rescale"],
        shape=info["shape"],
    )
    # Diferencia val_min y val_max de pixel
    rango = info["rango"]

    datos_train = preprocesado.datos_train(batch_size)
    datos_val = preprocesado.datos_val(batch_size)

    if loss == "mssim":
        loss_func = losses.mssim(rango)
    elif loss == "l2":
        loss_func = losses.l2_loss

    print("Preparando para entrenamiento...")
    autoencoder.compile(
        optimizer=Adam(learning_rate=lr), 
        loss=loss_func,
        metrics=[metrics.metrics(rango)],
        )

    if parametros.VERBOSE:
        autoencoder.summary()

    print(f"Iniciando entrenamiento para {epochs} epochs y learning rate inicial {lr}. Decay : {parametros.DECAY} | Step : {parametros.STEP}")
    hist = autoencoder.fit(
        datos_train, 
        validation_data=datos_val, 
        validation_steps= datos_val.samples // batch_size,
        epochs=epochs, 
        steps_per_epoch= datos_train.samples // batch_size, 
        verbose=parametros.VERBOSE, 
        callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler)],
    )

    tiempo = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    dir_guardado = os.path.join(
        os.getcwd(),
        "modelos_guardados",
        tipo, 
        loss,
        tiempo,
    )

    if not os.path.isdir(dir_guardado):
        os.makedirs(dir_guardado)

    print("Guardando historial..")
    keys = ["loss", "val_loss", "mssim", "val_mssim"]
    hist_dict = dict((key, hist.history[key]) for key in keys)
    hist_dict["epochs"] = list(range(1, epochs+1))
    hist_df = pd.DataFrame(hist_dict)
    archivo = os.path.join(dir_guardado, "hist.csv")
    with open(archivo, "w") as archivo_df:
        hist_df.to_csv(archivo_df)
    print("Historial guardado!\n")

    print("Guardando gráficas...")
    # Obtenemos datos
    with plt.style.context("seaborn-darkgrid"):
        fig1 = hist_df.plot(x="epochs", y=["loss", "val_loss"]).get_figure()
        plt.title("Gráfica loss")
        plt.show()
        plt.close()
        fig1.savefig(os.path.join(dir_guardado, "loss_plot.png"))
    with plt.style.context("seaborn-darkgrid"):
        fig2 = hist_df.plot(x="epochs", y=["mssim", "val_mssim"]).get_figure()
        plt.title("Gráfica precisión")
        plt.show()
        plt.close()
        fig2.savefig(os.path.join(dir_guardado, "precision_plot.png"))
    print("Gráficas guardadas!\n")
    
    

    print("Guardando modelo...")
    guardar(dir_guardado, dir_datos, datos_train, datos_val, autoencoder, tipo, batch_size, epochs, loss)
    tf.keras.utils.plot_model(autoencoder, to_file=os.path.join(dir_guardado, "modelo.png"), show_shapes=True)
    print("Modelo guardado!\n")

    print(f"Todo guardado en {dir_guardado}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="-d : directorio de datos | -t : tipo de modelo (normal o resnet) | -l : tipo de loss")

    parser.add_argument(
        "-d",
        "--dir-datos",
        type=str,
        required=True,
        metavar="",
    )

    parser.add_argument(
        "-t",
        "--tipo",
        type=str,
        required=False,
        metavar="",
        choices=["normal", "resnet"],
        default="normal",
    )

    parser.add_argument(
        "-l", 
        "--loss",
        type=str, 
        required=False,
        metavar="",
        choices = ["mssim", "l2"],
        default="mssim",
    )
    
    args = parser.parse_args()

    if tf.test.is_gpu_available():
        print("GPU detectado...")


    gpus = tf.config.experimental.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(gpus[0], True)

    print("Verstion tf : {} || Vesion keras : {}".format(tf.__version__, keras.__version__))
    # Ejecutamos
    main(args)