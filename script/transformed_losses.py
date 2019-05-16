

import sys
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, \
                UpSampling1D, Conv1D, ZeroPadding1D, Add, \
                BatchNormalization, Activation, LeakyReLU, \
                Dropout, Reshape, Embedding, Flatten, \
                Concatenate, Lambda, Conv2D

import keras.backend as K
import tensorflow as tf


from spectrogram_extractor import get_spectrogram_extractor
from data_utils import get_td_warper



def get_transformer(params):
    assert 'name' in params
    name = params['name']
    if name == 'identity':
        transformer = get_identity_map()

    elif name == 'td_warped':
        transformer = get_td_warper()
        

    elif name == 'spectrogram':
        assert params['normalisation'] != 'freq_global_norm'
        transformer = get_spectrogram_extractor(normalise=params['normalisation'], \
                dft_window=params['window'], n_hop=params['hop'], \
                decibel_melgram=params['decibel'], spectrogram_type='STFT', \
                power=params['power'])
    elif name == 'melspectrogram':
        assert params['normalisation'] != 'freq_global_norm'
        transformer = get_spectrogram_extractor(n_mels=params['ndim'], normalise=params['normalisation'], \
                dft_window=params['window'], n_hop=params['hop'], \
                decibel_melgram=params['decibel'], spectrogram_type='MelSTFT', \
                power=params['power'])

    elif name == 'spectrogram_patch_std':
        transformer = get_spectral_patch_std_net(params)        

    elif name == 'stoi' :
        assert params['normalisation'] == 'none'
        transformer = get_spectrogram_extractor(normalise=params['normalisation'], \
                dft_window=params['window'], n_hop=params['hop'], spectrogram_type='STOIgram')
    
    else:
        sys.exit('unknown transformation name: %s'%(params['name']))
    return transformer


def get_identity_map():
    inputs = Input(shape=(None, 1))      ### time x 1
    outputs = Lambda(lambda x: x)(inputs)    ### identity map
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model



def get_spectral_patch_std_net(params):
    spectrogram_extractor = get_spectrogram_extractor(normalise=params['normalisation'], \
        dft_window=params['window'], n_hop=params['hop'], \
        decibel_melgram=params['decibel'], spectrogram_type='STFT', \
        power=params['power'])

    chunksize = params['n_samples_in'] / params['hop']
    nin = (dft_window / 2) + 1
    (time, freq) = params['patch_size']
    time = min(time, chunksize)
    freq = min(freq, nin)
    patch_size = (time, freq)

    inputs = Input(shape=(None,nin))
    data = Lambda(get_patches, arguments={'insize':(chunksize, nin), 'patch_size':patch_size})(inputs)
    data = Lambda(K.std, arguments={'axis': -1})(data) ## 
    model = Model(inputs=[inputs], outputs=[data])
    return model










def get_std(x, int_axis=-1):
    if int_axis == -1:
        std = K.std(x, axis=[2, 1, 0], keepdims=True)
    elif int_axis == (0,1):
        std = K.std(x, axis=[2], keepdims=True)
    elif int_axis in (0, 1, 2):
        all_dims = [0, 1, 2]
        del all_dims[int_axis]
        std = K.std(x, axis=all_dims, keepdims=True)
    else:
        sys.exit('vsvsdfv')
    return std




def get_std_net(nin, chunksize, axis='batch'):
    ### batch actually outputs 1 std per example:
    # Raw shape:  (16, 1000, 152)
    # Transformed:  (16, 1, 1)
    assert axis in ['batch', 'data_sample', 'feat', 'time']

    if axis == 'batch':
        int_axis = -1
    elif axis == 'time':
        int_axis = (0, 1)   ## in case of time, return 1 std per frame, not averaged over examples
    else:
        int_axis = ['data_sample', 'time', 'feat'].index(axis)
        
    assert int_axis in (-1, 0, 1, 2, (0,1)), 'invalid int_axis: ' + str(int_axis)
    


    inputs = Input(shape=(chunksize,nin))
    #std = Lambda(get_std, output_shape=get_std_output_shape)(inputs)
    std = Lambda(get_std, arguments={'int_axis':int_axis})(inputs)  # output_shape: Only relevant when using Theano
    
    model = Model(inputs=[inputs], outputs=[std])

    return model


def get_patches(data, insize=(100,60), patch_size=(3,3)):
    data = K.expand_dims(data, axis=-1) ## add extra dummy 'channel' dimension 
    data = K.reshape(data, shape=(-1,insize[0],insize[1],1)) # just a hack to force time dimensions to be known (for extract_image_patches)
    kernel_size = [1,patch_size[0],patch_size[1],1]     
    strides = [1,1,1,1]
    rates = [1,1,1,1]
    padding = 'SAME'
    out = tf.extract_image_patches(data, kernel_size, strides, rates, padding)
    return out



def get_patch_std_net(nin, chunksize, patch_size=(3,3)):
    ## use of tf extract_image_patches fails, I think due to timelength being unspecified at 
    ## input of model: https://github.com/tensorflow/tensorflow/issues/11651
    ##
    ## Get:
    # File "/disk/scratch/oliver/virtual_python/waffler/lib/python2.7/site-packages/tensorflow/python/ops/array_grad.py", line 755, in _ExtractImagePatchesGrad
    # rows_out = int(ceil(rows_in / stride_r))
    # TypeError: unsupported operand type(s) for /: 'NoneType' and 'long'
    #
    # Fixed by adding insize input to get_patches, to spell out time length explicitly

    '''
    patch_size = size in (time x freq)
    '''
    (time, freq) = patch_size
    time = min(time, chunksize)
    freq = min(freq, nin)
    patch_size = (time, freq)

    inputs = Input(shape=(None,nin))
    data = Lambda(get_patches, arguments={'insize':(chunksize, nin), 'patch_size':patch_size})(inputs)
    data = Lambda(K.std, arguments={'axis': -1})(data) ## 
    model = Model(inputs=[inputs], outputs=[data])
    return model








# https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1
def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def tf_cov_3D_OLD(x, nfeats=1):
    # As tf_cov, but flatten batches
    #a,b,c = x.get_shape()
    x_matrix = tf.reshape(x, (-1, nfeats)) ### flatten batch dim o we take the cov between features over time, regardless of example
    mean_x = tf.reduce_mean(x_matrix, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x_matrix), x_matrix)/tf.cast(tf.shape(x_matrix)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx

def tf_cov_3D(x_with_batch):
    return tf.map_fn(tf_cov, x_with_batch)



def get_covarnet(nin, chunksize):

    inputs = Input(shape=(chunksize,nin))
    cov = Lambda(tf_cov_3D)(inputs)  # output_shape: Only relevant when using Theano
    model = Model(inputs=[inputs], outputs=[cov])
    return model


def test_cov():
    # https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1
    data = np.array([[1., 4, 2], [5, 6, 24], [15, 1, 5], [7,3,8], [9,4,7]])

    with tf.Session() as sess:
        print(sess.run(tf_cov(tf.constant(data, dtype=tf.float32))))

    ## validating with numpy solution
    pc = np.cov(data.T, bias=True)
    print(pc)

    data_3D = np.array([[[10., 4, 2], [5, 6, 24], [15, 1, 5], [7,3,8], [9,4,7]], [[1., 4, 2], [5, 6, 24], [15, 1, 5], [7,3,8], [9,4,7]]])
    print data_3D.shape
    with tf.Session() as sess:
        print(sess.run(tf_cov_3D(tf.constant(data_3D, dtype=tf.float32))))




def test_patches():
    data = np.random.uniform(size=(16,1000,152,1))
    kernel_size = [1,10,5,1]                     ### gives output size: (16, 991, 148, 50)   -- 5 x 10 patches are put in the 'depth' dimension 
    strides = [1,1,1,1]
    rates = [1,1,1,1]
    padding = 'VALID'
    with tf.Session() as sess:
        out = sess.run(tf.extract_image_patches(tf.constant(data, dtype=tf.float32),
                                                        kernel_size, strides, rates, padding ))
        print out
        print out.shape



def test_patches2():
    # https://stackoverflow.com/questions/50980113/evaluate-a-function-in-a-sliding-window-with-keras
    import tensorflow as tf
    from keras import layers, models
    import keras.backend as K

    win_len = 20   # window length
    feat_len = 4   # features length

    def extract_patches(data):
        data = K.expand_dims(data, axis=3)
        patches = tf.extract_image_patches(data, ksizes=[1, win_len, feat_len, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        return patches

    target = layers.Input((None, feat_len))
    patches = layers.Lambda(extract_patches)(target)
    patches = layers.Reshape((-1, win_len, feat_len))(patches)

    model = models.Model([target], [patches])
    model.summary()


    import numpy as np

    # an array consisting of numbers 0 to 399 with shape (100, 4)
    a = np.arange(1*100*4*1).reshape(1, 100, 4)
    print(model.predict(a))


if __name__ == '__main__':
    #test_cov()
    test_patches2()
