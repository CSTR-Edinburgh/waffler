
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, \
                UpSampling1D, Conv1D, ZeroPadding1D, Add, \
                BatchNormalization, Activation, LeakyReLU, \
                Dropout, Reshape, Lambda, SeparableConv1D, Multiply
import keras.layers as layers 
import keras.backend as K
import keras.initializers as initializers 

import numpy as np




###### ----- some building blocks --------


def linear_transform(dim, activation=None):
    '''
    1x1 convolution, for time-distributed linear transform
    '''
    return Conv1D(dim, 1, strides=1, padding='same', dilation_rate=1, activation=activation, \
                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', \
                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, \
                  kernel_constraint=None, bias_constraint=None)

def conv_relu(dim, conv_width, padding='same', activation='relu', dilation_rate=1, depthwise_separable=False):
    if depthwise_separable:
        return  SeparableConv1D(dim, conv_width, strides=1, padding=padding, dilation_rate=dilation_rate, activation=activation, \
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', \
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, \
                 kernel_constraint=None, bias_constraint=None)

    else:
        return Conv1D(dim, conv_width, strides=1, padding=padding, dilation_rate=dilation_rate, activation=activation, \
                     use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', \
                     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, \
                     kernel_constraint=None, bias_constraint=None)


def residual_block(data, in_shape=(None,1), convlayers=3, convwidth=3, get_edge_int=False, padding='same', dilation_rate=1, depthwise_separable=False, activation='relu'):
    m,dim = in_shape  ## use dim for conv layers so that transformed and original data can be combined
    transformed_data = conv_relu(dim, convwidth, padding=padding, dilation_rate=dilation_rate, depthwise_separable=depthwise_separable)(data)
    for subsequent_layer in range(convlayers-1):
        transformed_data = conv_relu(dim, convwidth, padding=padding, dilation_rate=dilation_rate, depthwise_separable=depthwise_separable, activation=activation)(transformed_data)
    transformed_data = layers.add([data, transformed_data]) 
    extra_width = (convlayers * (convwidth-1)) # + 1
    edge_interference = 0 ## TODO: implement this?
    if get_edge_int:
        return transformed_data, extra_width, edge_interference
    else:
        return transformed_data, extra_width

def residual_block_b(data, in_shape=(None,1), convlayers=3, convwidth=3, get_edge_int=False, padding='same', dropout=0.0,  dilation_rate=1, batchnorm=True):
    '''
    As residual_block, but use leaky relu and add norm after every conv, optionally use dropout
    '''
    m,dim = in_shape  ## use dim for conv layers so that transformed and original data can be combined
    transformed_data = conv_relu(dim, convwidth, padding=padding, dilation_rate=dilation_rate)(data)
    if batchnorm:
        transformed_data = norm()(transformed_data)
    if dropout > 0.0:
        transformed_data = Dropout(dropout)(transformed_data)
    for subsequent_layer in range(convlayers-1):
        transformed_data = conv_relu(dim, convwidth, padding=padding, activation=None, dilation_rate=dilation_rate)(transformed_data)
        transformed_data = LeakyReLU(alpha=0.3)(transformed_data)
        if batchnorm:
            transformed_data = norm()(transformed_data)
        if dropout > 0.0:
            transformed_data = Dropout(dropout)(transformed_data)        
    transformed_data = layers.add([data, transformed_data]) 
    extra_width = (convlayers * (convwidth-1)) # + 1
    edge_interference = 0 ## TODO: implement this?
    if get_edge_int:
        return transformed_data, extra_width, edge_interference
    else:
        return transformed_data, extra_width


def highway_block(data, prefix='highway_1', channels=16, convwidth=3, dilation_rate=1, activation='relu', initializer='glorot_uniform', gate_bias=-3.0):
    n = prefix
    transformed_data = Conv1D(channels, convwidth, dilation_rate=dilation_rate, activation=activation,\
                 kernel_initializer=initializer, padding='same', name='%s_trans'%(n))(data)
    transform_gate = Conv1D(channels, convwidth, dilation_rate=dilation_rate, activation='sigmoid',\
                 kernel_initializer=initializer, bias_initializer=initializers.Constant(value=gate_bias), padding='same', name='%s_Tgate'%(n))(data)
    transformed_data = Multiply(name='%s_Tgater'%(n))([transform_gate, transformed_data]) 
    carry_gate = Lambda(lambda x: 1.0 - x, name='%s_Cgate'%(n))(transform_gate)
    carried_data = Multiply(name='%s_Cgater'%(n))([carry_gate, data]) 
    return Add(name='%s_adder'%(n))([transformed_data, carried_data])

def norm():
    return BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, \
                 beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', \
                 moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, \
                 beta_constraint=None, gamma_constraint=None)    




###### ----- architectures built on the basic blocks --------
def Q03(vocdim, layers_per_block=[3,3,3,3,1], widths=[9,9,9,9,3], dilations=[10,5,2,1,1], \
            hop=200, observation_dim=1, convchannels=64, do_norm=True, depthwise_separable=False, \
            activation='relu', output_activation='linear'):

    assert len(layers_per_block) == len(widths) == len(dilations)
    nblocks = len(layers_per_block)
    if type(convchannels) == list:
        assert len(convchannels) == nblocks
    else:
        assert type(convchannels) == int 
        convchannels = [convchannels] * nblocks

    ## check input data:
    for seq in [layers_per_block, widths, dilations, convchannels]:
        for element in seq:
            assert type(element) == int and element > 0

    melspec_inputs = Input(shape=(None, vocdim, 1))   ##  (B x T x F x 1), as returned by spectrogram extractor
    excitation_inputs = Input(shape=(None, 1))        ##  (B x T x 1)
    noise_inputs = Input(shape=(None, 1))             ##  (B x T x 1)
    melspec_data = Lambda(lambda x: x[:,:,:,0])(melspec_inputs) ## (B x T x F x 1) -> (B x T x F)

    melspec_data = UpSampling1D(size=hop)(melspec_data)

    indata = [melspec_data]
    indata.append(excitation_inputs)
    indata.append(noise_inputs)
    data = layers.concatenate(indata, axis=-1)

    prev_channels = -1 ## negative initial value means we'll always linearly project before first res block
    for (block_number, (l,w,d,channels)) in enumerate(zip(layers_per_block, widths, dilations, convchannels)):
        if prev_channels != channels:
            ## linearly transform all input data
            data = linear_transform(channels)(data)   

        data, extra_width = residual_block(data, in_shape=(None, channels), convlayers=l, convwidth=w, dilation_rate=d, depthwise_separable=depthwise_separable, activation=activation)
        if do_norm:
            data = norm()(data)    
        prev_channels = channels

    prediction = linear_transform(1, activation=output_activation)(data)

    model = Model(inputs=[melspec_inputs, excitation_inputs, noise_inputs], outputs=[prediction])
    model.summary()    
    return model
