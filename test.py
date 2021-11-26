import os
import argparse
import json

import tensorflow as tf
import numpy as np
import pandas as pd

import processing
import parametros

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt

import sys

def barra_progreso(iter, primer_char="", total=60, file=sys.stdout):
    n = len(iter)
    def show(j):
        x = int(total*j/n)
        file.write("%s[%s%s] %i/%i\r" % (primer_char, "#"*x, "."*(total-x), j, n))
        file.flush()        
    show(0)
    for i, item in enumerate(iter):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def obtener_tipo(archivos):
    label_true = [0 if "good" in imagen.split("/") else 1 for imagen in archivos]
    return label_true

def defectuosa(areas, min_area):
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    else:
        return 0

def prediccion_defecto(resmaps, min_area, baremo):
    
    # Obtenemos de la imagen de diferencias (resmap) las partes con diferencia mayor al baremo 
    imagen_anomalia = resmaps > baremo

    _, total_areas = processing.clasificar_imagenes(imagen_anomalia)

    label_pred = [defectuosa(areas, min_area) for areas in total_areas]
    
    return label_pred

def main(args):

    dir_modelo = args.dirmodelo

    # DATOS
    model, info = processing.cargar_modelo(dir_modelo)

    dir_datos = info["datos"]["dir_datos"]
    nombre_modelo = info["modelo"]["tipo"]
    loss = info["modelo"]["loss"]
    rescale = info["preprocesamiento"]["rescale"]
    shape = info["preprocesamiento"]["shape"]
    val_min = info["preprocesamiento"]["val_min"]
    val_max = info["preprocesamiento"]["val_max"]
    num_imagenes_val = info["datos"]["num_imagenes_validacion"]



    preprocesado = processing.GeneradorDatos(
        dir_datos=dir_datos, 
        rescale=rescale, 
        shape=shape,)


    num_imgs_test = preprocesado.num_imagenes_test()
    datos_test = preprocesado.datos_test(batch_size=num_imgs_test, shuffle=False)

    # Como datos_test es un DirectoryIterator, obtenemos las imagenes de test del generador
    img_test = datos_test.next()[0]

    # Nombres de las imagenes
    archivos = datos_test.filenames

    
    test_eval = model.evaluate(datos_test)

    print("Loss Test: ", test_eval[0])
    print("Precisión: ", test_eval[1])

    prediccion = model.predict(img_test)
    
    tensor_imgs = processing.ArrayImagen(
        entrada=img_test, 
        predicciones=prediccion,
        val_min=val_min,
        val_max=val_max,
        metodo=parametros.METODO,
        nombre_imagenes=archivos
    )


    clase_real = obtener_tipo(archivos)

    if parametros.METODO ="l2":
        start_baremo = 0.1
        stop_baremo = 0.5
        step_baremo = parametros.STEP_BAREMO
    elif parametros.METODO = "ssim":
        start_baremo = 0.5
        stop_baremo = 1
        step_baremo = parametros.STEP_BAREMO

    mejor_puntuacion = 0
    puntuación prev = 0
    mejor_baremo = None
    mejor_area_min = None
    mejor_falsos_positivos = None
    mejor_falsos_negativos = None
    mejor_rda = None
    mejor_rnda = None
    for i in barra_progreso(range(100), "Calculando mejores parámetros para detectar anomalías: ", 40):
        for i in range(parametros.ITER_TEST)
            for area_min in list(np.arange(start=parametros.START_AREA_MIN, stop=parametros.STOP_AREA_MIN, step=parametros.STEP_AREA_MIN)):
                for baremo in list(np.arange(start=start_baremo, stop=stop_baremo, step=step_baremo)):

                        clase_prediccion = prediccion_defecto(
                            resmaps=tensor_imgs.resmaps,
                            min_area=area_min,
                            baremo=baremo,
                        )
                        
                        # Computamos los parámetros para calcular la puntuación
                        ratio_defectos_acertados, falsos_positivos, falsos_negativos, ratio_no_defectos_acertados = confusion_matrix(clase_real, clase_prediccion, normalize="true").ravel()
                        puntuacion = (ratio_defectos_acertados + ratio_no_defectos_acertados)/2
                        

                        # Si es mejor, la guardamos
                        if puntuacion > mejor_puntuacion:
                            mejor_puntuacion = puntuacion
                            mejor_baremo = baremo
                            mejor_area_min = area_min
                            mejor_falsos_positivos = falsos_positivos
                            mejor_falsos_negativos = falsos_negativos
                            mejor_rda = ratio_defectos_acertados
                            mejor_rnda = ratio_no_defectos_acertados
                        
        
            # Acotamos el baremo y reducimos el step por un decimal para aumentar precisión
            start_baremo = mejor_baremo - step_baremo
            stop_baremo = mejor_baremo + step.baremo
            step_baremo = step_baremo*0.1
    
    resultados = {
        "puntuacion" : str(mejor_puntuacion),
        "ratio_defectos_acertados" : str(mejor_rda),
        "ratio_no_defectos_acertados" : str(mejor_rnda),
        "falsos positivos" : str(mejor_falsos_positivos), 
        "falsos negativos" : str(mejor_falsos_negativos),
        "método" : parametros.METODO,
        "baremo" : str(mejor_baremo), 
        "area min" : str(mejor_area_min),
    }

    print(f"Para BAREMO :  {mejor_baremo} y AREA MÍNIMA : {mejor_area_min} se obtienen los siguientes resultados...")
    print("PUNTUACIÓN [(RATIO DEFECTOS ACERTADOS + RATIO NO DEFECTOS ACERTADOS) / 2)] : {}".format(resultados["puntuacion"]))
    print("Ratio defectos acertados : {}".format(resultados["ratio_defectos_acertados"]))
    print("Ratio no defectos acertados : {}".format(resultados["ratio_no_defectos_acertados"]))


    print("\nGuardando datos...")
    
    dir_guardado = os.path.join(dir_modelo, "test")
    
    if not os.path.isdir(dir_guardado):
        os.makedirs(dir_guardado)

    with open(os.path.join(dir_guardado, "resultados.json"), "w") as archivo:
        json.dump(resultados, archivo, indent=4)

    print(f"Resultados guardados en {dir_guardado}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description= "Argumentos posibles\n -d : directorio del modelo guardado (archvivo .hdf5)"
        )
    
    parser.add_argument(
        "-d", "--dirmodelo", type=str, required=True, metavar="", help="directorio del modelo"
    )

    args = parser.parse_args()

    main(args)

