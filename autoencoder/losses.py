import tensorflow as tf

def ssim(rango):
    def loss(imgs, pred):
        return 1 - tf.image.ssim(imgs, pred, rango)

    return loss

def mssim(rango):
    def loss(imgs, pred):
        return 1 - tf.image.ssim_multiscale(imgs, pred, rango)

    return loss


def l2_loss(imgs, pred):
    return tf.nn.l2_loss(imgs - pred)