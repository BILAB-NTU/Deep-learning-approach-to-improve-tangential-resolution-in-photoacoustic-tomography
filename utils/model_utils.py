
from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2DTranspose





########################################################################################################################
'''MODEL UTILITIES:'''
########################################################################################################################
def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    ############################################

    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation='linear', kernel_initializer='glorot_normal')
    shortcut = out
    out = DownSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return [out, shortcut]
########################################################################################################################

def RESBridgeBlock(input, filters,no_of_resnetblocks,kernel_size, padding, activation, kernel_initializer):
    ############################################
    f_in = filters

    out = input
    for i in range(no_of_resnetblocks):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=kernel_size, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Add()([out, shortcut])

    out = UpSample(out, f_in*2, kernel_size, strides=2, padding=padding,activation=activation, kernel_initializer=kernel_initializer)

    ############################################
    return out
########################################################################################################################


def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    ############################################
    out = Conv2D_BatchNorm(input, filters=filters // 2, kernel_size=1, strides=1, padding=padding,
                           activation='linear', kernel_initializer=kernel_initializer)
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation='linear', kernel_initializer='glorot_normal')
    out = UpSample(out, filters , kernel_size, strides=2, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)
    ############################################
    return out



########################################################################################################################
'''FUNCTIONS FOR MODEL UTILITIES:'''
########################################################################################################################


def Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding='same', activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                 activation=activation, kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    return out


def Conv2D_Transpose_BatchNorm(input, filters, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal'):

    out = Conv2DTranspose(filters, kernel_size, strides=2, padding=padding, activation=activation, kernel_initializer=kernel_initializer)(input)
    out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
                             scale=True, beta_initializer='zeros', gamma_initializer='ones',
                             moving_mean_initializer='zeros', moving_variance_initializer='ones',
                             beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                             gamma_constraint=None)(out)
    return out


def DownSample(input, filters, kernel_size=3, strides=2, padding='same', activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_BatchNorm(out, filters, kernel_size, strides=strides, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    return out


def UpSample(input, filters, kernel_size=3, strides=2, padding='same',
             activation='linear', kernel_initializer='glorot_normal'):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding, activation=activation, kernel_initializer=kernel_initializer)

    out = Conv2D_Transpose_BatchNorm(out, filters // 2, kernel_size, strides=strides, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    return out


########################################################################################################################
'''FULLY DENSE BLOCK:'''
########################################################################################################################
def FD_Block(input, f_in, f_out, k, kernel_size=3, padding='same', activation='linear', kernel_initializer='glorot_normal'):
    out = input
    for i in range(f_in, f_out, k):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Conv2D_BatchNorm(out, filters=k, kernel_size=kernel_size, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Concatenate()([out, shortcut])
    return out


########################################################################################################################