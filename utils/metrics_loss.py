
import tensorflow as tf
from utils import metrics_loss


##################################################################################################################################
'''
MODEL UTILS:
'''
##################################################################################################################################


def normalize(tensor):

    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), 
                                 tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(metrics_loss.normalize, y_true)
    y_pred_norm = tf.map_fn(metrics_loss.normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR

def SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(metrics_loss.normalize, y_true)
    y_pred_norm = tf.map_fn(metrics_loss.normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,
                         filter_sigma=1.5,k1=0.01,k2=0.03)
    return SSIM
def KLDivergence(y_true, y_pred):
    return tf.losses.KLDivergence()(y_true, y_pred)

def SavingMetric(y_true, y_pred):
    SSIM = metrics_loss.SSIM(y_true, y_pred)
    PSNR = metrics_loss.PSNR(y_true, y_pred)
    SSIM_norm = 1 - SSIM
    PSNR_norm = (40 - PSNR)/275
    loss = SSIM_norm + PSNR_norm
    return loss

def FFT_mag(input):
    real = input
    imag = tf.zeros_like(input)
    out = tf.abs(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0]))
    return out

def model_loss(B1=1.0, B2=0.01, B3=0.0, B4=0.0):
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        if tf.executing_eagerly():
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred).numpy()
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred))).numpy()
        else:
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
            MaxAbsDiff_Loss = tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred)))
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)
        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss + B4*MaxAbsDiff_Loss
        return loss
    return loss_func

def model_loss_experimental(B1=1.0, B2=0.0, B3=0.0):
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        if tf.executing_eagerly():
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred).numpy()
            saving_metric = metrics_loss.SavingMetric(y_true, y_pred).numpy()
        else:
            MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
            F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
            saving_metric = metrics_loss.SavingMetric(y_true, y_pred)
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)

        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss + B3*saving_metric
        return loss
    return loss_func