import numpy as np
from numpy import load
from matplotlib import pyplot
from gooey import Gooey, GooeyParser

@Gooey(program_name='View training, testing and validation data')
def main():
    parser = GooeyParser()
    parser.add_argument('training_dataset', help='train file name with .npz extension', type=str, widget='FileChooser')
    args = parser.parse_args()
    return args.training_dataset


def plot(train_filename):
    data = load(train_filename)
    src_images, tar_images = data['arr_0'], data['arr_1']
    src_images = np.reshape(src_images, (src_images.shape[0], 128, 128))
    tar_images = np.reshape(tar_images, (tar_images.shape[0], 128, 128))
    print('Loaded:', src_images.shape, tar_images.shape)
    # plot source images
    n_samples = 3
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(src_images[i], cmap='gist_yarg')
    # plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(tar_images[i], cmap='gist_yarg')
    pyplot.show()

if __name__=='__main__':
    training_dataset= main()
    plot(training_dataset)

