

import keras.backend as K

from keras.models import Model
from keras.layers import Input, Lambda 

import numpy as np
import scipy

def trim_to_nearest(data, to_nearest):
    to_nearest = int(to_nearest)
    length = (len(data) / to_nearest) * to_nearest
    return data[:length]

def lin2log(x):
    mu=256.0
    return K.sign(x) * K.log(1.0 + mu*K.abs(x))/K.log(1.0 + mu)    

def get_td_warper():
    inputs = Input(shape=(None, 1))      ### time x 1
    warped_data = Lambda(lin2log)(inputs)  
    model = Model(inputs=inputs, outputs=warped_data)
    model.summary()
    return model

def tweak_batch(data_batch, input_spectrogram_extractor, config, output_transformers):

    '''
    Take raw data returned by provider and tweak it into network inputs/outputs.
    Tweaking means e.g. normalisation, spectrogram extraction, converting to fp16

    TODO: decide where to put this work

    input dimensions:
        data_batch = (wav, exc)  ## typically
        
        wav: batch of waveform fragments, batchsize x fragment length (B x T)  (e.g. 8 x 16000)
        exc: batch of pulse excitation fragments (same dimensions as wav)
    '''
    noise_input = config.get('add_noise', False)
    noise_std = config.get('noise_std', 1.0)   
    hop = config.get('n_hop', 256)    

    warped_td_loss = config.get('warped_td_loss', False)

    halfrange = (2**16) / 2.0

    (wav, exc) = data_batch
    
    wav = wav / halfrange        #### TODO: move reshaping and scaling to provider?
    exc = exc / halfrange      

    if wav.shape[1] % hop != 0:
        to_nearest = int(hop)
        length = (wav.shape[1] / to_nearest) * to_nearest    
        if 0: print 'Warning: trim data from %s to %s so compatible with spectrogram extractor hop %s'%(wav.shape[1], length, hop)    
        wav = wav[:,:length]
        exc = exc[:,:length]

    wav = np.expand_dims(wav, -1)   ## (B x T) -> (B x T x 1) 
    exc = np.expand_dims(exc, -1)   ## (B x T) -> (B x T x 1) 

    in_melspec = input_spectrogram_extractor.predict(wav)  ## wav: (B x T x 1) ; in_melspec: (B x T x F x 1)

    inputs = [in_melspec, exc]

    if noise_input:
        noise = np.random.normal(0,noise_std,size=exc.shape)
        inputs.append(noise)

    target = [transformer.predict(wav) for transformer in output_transformers]

    return (inputs, target)  



