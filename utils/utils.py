import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def data_gen(filename):
    path = tf.keras.utils.get_file(filename,origin=None)
    with np.load(path) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images
def data_gen_exp(filename):
    path = tf.keras.utils.get_file(filename,origin=None)
    with np.load(path) as data:
        src_images = data['arr_0']
    return src_images

def plot_history(history,metric,valmetric):
    offset=0
    
    data1 = history.history[metric][offset:]
    data2 = history.history[valmetric][offset:]
    epochs = range(offset, len(data1) + offset)
    plt.plot(epochs, data1)
    plt.plot(epochs, data2)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(metric)
    plt.close()

