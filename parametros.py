# import numpy as np

VERBOSE = True

""" PROCESSING """
# Entreno
INIT_LR = 0.001
DECAY = 0.5
STEP = 5
EPOCHS = 25
BATCH_SIZE = 8

# Preprocessing
VAL_SPLIT = 0.2
ANGULO = 75
DESPLAZAMIENTO_HORIZONTAL = 0.3
DESPLAZAMIENTO_VERTICAL = 0.3
LLENADO = "nearest"
BRILLO = [0.75, 1.25]


# Post processing - TEST
METODO = "ssim"
START_AREA_MIN = 0
STOP_AREA_MIN = 20
STEP_AREA_MIN = 5
STEP_BAREMO = 0.01

# Cuantos decimales de precisón queremos para le baremo
ITER_TEST = 5

# MSSIM/L2
# BAREMO 0.1215384
# AREA 0.4

# L2/L2

# Modelos
# NORMAL
info_normal = {
    "loss" : "",
    "rescale" : 1.0 / 255,
    "shape" : (256, 256),
    "val_min" : 0.0,
    "val_max" : 1.0,
    "rango" : 1.0 - 0.0
}

# RESNET
info_resnet = {
    "loss" : "",
    "rescale" : 1.0 / 255,
    "shape" : (256, 256),
    "val_min" : 0.0, 
    "val_max" : 1.0,
    "rango" : 1.0 - 0.0,
}
