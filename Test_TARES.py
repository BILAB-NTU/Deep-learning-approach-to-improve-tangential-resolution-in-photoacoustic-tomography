import os
import numpy as np
import tensorflow as tf
from utils.utils import data_gen
from matplotlib import pyplot as plt
from utils.metrics_loss import model_loss, PSNR, SSIM, KLDivergence, SavingMetric
from gooey import Gooey, GooeyParser


@Gooey(program_name='TARES test')
def main():
    parser = GooeyParser()
    parser.add_argument('test_dir', help='test file name with .npz extension',type=str, widget='FileChooser',default='')
    parser.add_argument('model_dir', help='directory where the model is saved', type=str, widget='DirChooser', default='')
    parser.add_argument('num_samples', help='number of samples to plot ',type=int,  default= 3)
    parser.add_argument('image_number', help='range of images ', type=int, default=0)
    parser.add_argument('fig_name',  type=str, default='testimage')
    parser.add_argument('save_fig', help='If you want to save figure input True', default=False)
    args = parser.parse_args()
    return args.test_dir,args.model_dir,args.num_samples,args.image_number,args.fig_name,args.save_fig




def model_load(test_filename,model_dir,num_samples,im_no,fig_name,save_fig):
    predictions_test = list()
    src_test , tar_test = data_gen(test_filename)
    print(len(src_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((src_test))
    test_dataset = test_dataset.batch(1)
    filename = sorted(os.listdir(model_dir), key = lambda x : int(x.partition('h_')[2].partition('-S')[0]))[-1]
    directory = os.path.join(os.getcwd(), model_dir)
    directory = os.path.join(directory, filename)
    saved_model = tf.keras.models.load_model(directory, custom_objects={'model_loss':model_loss,'loss_func':model_loss(B1=0.99,B2=0.01),'PSNR':PSNR, 'SSIM':SSIM, 'KLDivergence':KLDivergence,'SavingMetric':SavingMetric})
    print('Done Loading Best Model(' + filename + ') from: ' + model_dir)
    for element in test_dataset.as_numpy_iterator():
        predictions_curr = saved_model.predict(element, steps = 1)
        predictions_test.append(predictions_curr)
    [predictions_test] = [np.asarray(predictions_test)]
    predictions = np.reshape(predictions_test, (predictions_test.shape[0],128, 128))
    src_images = np.reshape(src_test, (src_test.shape[0],128, 128))
    tar_images = np.reshape(tar_test, (tar_test.shape[0],128, 128))
    for i in range(num_samples):
        plt.subplot(3, num_samples, 1 +  i)
        plt.axis('off')
        plt.imshow(src_images[i+im_no],cmap='gist_yarg')
        plt.title('input')
        plt.subplot(3, num_samples, 1 +num_samples+ i)
        plt.axis('off')
        plt.imshow(predictions[i+im_no],cmap='gist_yarg')
        plt.title('predicted')
        plt.subplot(3, num_samples, 1 + num_samples*2  + i)
        plt.axis('off')
        plt.imshow(tar_images[i+im_no],cmap='gist_yarg')
        plt.title('ground truth')
    if save_fig is True:
        plt.savefig(fig_name+'.jpg',dpi=150)
    plt.show()


if __name__=='__main__':
    test_dir, model_dir,num_samples, image_number, fig_name, save_fig = main()
    model_load(test_dir, model_dir, num_samples, image_number, fig_name, save_fig)
