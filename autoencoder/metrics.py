import tensorflow as tf
import tensorflow.keras.backend as K

# Usamos un decorator para que la función tome dos parametros, pero usando tres
def metrics(rango):
    def mssim(imgs, pred):
        return K.mean(
            tf.image.ssim_multiscale(imgs, pred, rango), axis=-1
            )

    return mssim