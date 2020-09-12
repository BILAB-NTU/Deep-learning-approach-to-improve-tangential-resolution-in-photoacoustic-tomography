from tensorflow.keras import Model
from tensorflow.keras.layers import  Conv2D,  Add, Input, Concatenate
from utils.model_utils import  Conv2D_BatchNorm,DownBlock,RESBridgeBlock,UpBlock,FD_Block


########################################################################################################################
'''Modified Dense UNet with residual blocks'''
########################################################################################################################

def Modified_D_UNet(input, filters=32, no_of_resnetblocks =2,kernel_size=3, padding='same',activation='relu', kernel_initializer='glorot_normal'):
    shortcut1_1 = input
    out = Conv2D_BatchNorm(input, filters, kernel_size=3, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    [out, shortcut1_2] = DownBlock(out, filters * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut2_1] = DownBlock(out, filters * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut3_1] = DownBlock(out, filters * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    [out, shortcut4_1] = DownBlock(out, filters * 2 * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = RESBridgeBlock(out,filters * 2 * 2 * 2 * 2 ,no_of_resnetblocks,kernel_size, padding, activation , kernel_initializer )
    out = Concatenate()([out, shortcut4_1])
    print('Conc_out: ' + str(out.shape))
    out = UpBlock(out, filters * 2 * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut3_1])
    print('Conc_out: ' + str(out.shape))
    out = UpBlock(out, filters * 2 * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut2_1])
    print('Conc_out: ' + str(out.shape))
    out = UpBlock(out, filters * 2 * 2, kernel_size, padding, activation, kernel_initializer)
    out = Concatenate()([out, shortcut1_2])
    print('Conc_out: ' + str(out.shape))
    out = FD_Block(out, f_in=filters, f_out=filters * 2, k=filters // 4, kernel_size=3, padding='same',activation='linear', kernel_initializer='glorot_normal')
    out = Conv2D(filters=1, kernel_size=1,strides=1, padding=padding,activation='linear', kernel_initializer=kernel_initializer)(out)
    out = Add()([out, shortcut1_1])
    return out


def getModel(input_shape, filters, kernel_size,no_of_resnetblocks, padding='same',activation='relu', kernel_initializer='glorot_normal'):
    model_inputs = Input(shape=input_shape, name='img')
    model_outputs = Modified_D_UNet(model_inputs, filters=filters,no_of_resnetblocks = no_of_resnetblocks, kernel_size=kernel_size, padding=padding,activation=activation, kernel_initializer=kernel_initializer)
    model = Model(model_inputs, model_outputs, name='FD-UNet_Model')

    return model