import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from utils.utils import data_gen,plot_history
from model.TARES_network import getModel
from utils.metrics_loss import model_loss, PSNR,SSIM,KLDivergence,SavingMetric
from gooey import Gooey, GooeyParser

@Gooey(program_name='TARES Training')
def main():
    parser = GooeyParser()
    parser.add_argument('training_dataset', help='train file name with .npz extension',type=str, widget='FileChooser')
    parser.add_argument('validation_dataset', help='validation file name with .npz extension', type=str, widget='FileChooser')
    parser.add_argument('Model_directory', help='name of the directory to store the trained model', type=str, default='Trained_model')
    parser.add_argument('Epochs', help='Number of epochs ', type=int, default=100)
    args = parser.parse_args()
    return args.training_dataset,args.validation_dataset,args.Model_directory,args.Epochs

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def training(Epochs,train_filename,validation_filename,model_dir):
    B1 = 1.0
    B2 = 0.01
    Filters = 32
    Kernel_size = 3
    Batch_size = 2
    Number_of_RES_blocks = 9
    Activation = 'selu'
    Input_image_shape = (128, 128, 1)
    delete_previous = True
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    elif delete_previous:
        shutil.rmtree(model_dir)
        os.mkdir(model_dir)
    bestmodel_dir = model_dir
    if not os.path.exists(bestmodel_dir):
        os.mkdir(bestmodel_dir)
    elif delete_previous:
        shutil.rmtree(bestmodel_dir)
        os.mkdir(bestmodel_dir)
    src_train, tar_train = data_gen(train_filename)
    src_valid, tar_valid = data_gen(validation_filename)
    print(len(src_train)), print(len(src_valid))
    train_dataset = tf.data.Dataset.from_tensor_slices((src_train, tar_train))
    train_dataset = train_dataset.repeat(-1)
    valid_dataset = tf.data.Dataset.from_tensor_slices((src_valid, tar_valid))
    valid_dataset = valid_dataset.repeat(-1)
    train_dataset = train_dataset.batch(Batch_size)
    valid_dataset = valid_dataset.batch(Batch_size)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')
    unet_model = getModel(input_shape=Input_image_shape, filters=Filters, no_of_resnetblocks=Number_of_RES_blocks,kernel_size=Kernel_size, activation=Activation)
    unet_model.compile(optimizer=opt, loss=model_loss(B1, B2),metrics=['mean_absolute_error', 'mean_squared_error', KLDivergence, SavingMetric, PSNR, SSIM])
    bestmodel_callbacks = ModelCheckpoint(filepath=os.path.join(bestmodel_dir,'saved_model.epoch_{epoch:02d}-SSIM_{val_SSIM:.5f}-PSNR_{val_PSNR:.5f}-metric_{val_SavingMetric:.5f}.h5'),monitor='val_SavingMetric', verbose=0, save_best_only=True,save_weights_only=False, mode='min', save_freq='epoch')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_SavingMetric', factor=0.5, patience=10, verbose=1, mode='min',min_lr=0.000001, epsilon=1e-04, )
    history = unet_model.fit(train_dataset, steps_per_epoch=np.ceil(len(src_train) / Batch_size), epochs=Epochs,callbacks=[bestmodel_callbacks, reduce_lr_loss], validation_data=valid_dataset,validation_steps=np.ceil(len(src_valid) / Batch_size), max_queue_size=256, shuffle=True,verbose=2)
    plot_history(history, 'loss', 'val_loss')
    plot_history(history, 'mean_absolute_error', 'val_mean_absolute_error')
    plot_history(history, 'mean_squared_error', 'val_mean_squared_error')
    plot_history(history, 'KLDivergence', 'val_KLDivergence')
    plot_history(history, 'PSNR', 'val_PSNR')
    plot_history(history, 'SSIM', 'val_SSIM')
    plot_history(history, 'SavingMetric', 'val_SavingMetric')

if __name__=='__main__':
    training_dataset,validation_dataset,Model_directory,Epochs= main()
    training(Epochs,training_dataset,validation_dataset,Model_directory)


