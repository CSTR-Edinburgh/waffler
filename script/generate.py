import os
import timeit
from argparse import ArgumentParser
import soundfile
import h5py

import numpy as np
import scipy

from keras.models import load_model, Model
from keras import layers

from namelib import get_model_dir_name, get_synth_dir_name, get_testset_names
from libutil import safe_makedir, load_config, write_wave, write_textlist_to_hdf
from spectrogram_extractor import get_spectrogram_extractor
from data_utils import tweak_batch, trim_to_nearest ## TODO: place?
from pitchmarks_to_excitation import get_epoch_position_features


### TODO: merge with pm reading function
def read_est_file(est_file):

    with open(est_file) as fid:
        header_size = 1 # init
        for line in fid:
            if line == 'EST_Header_End\n':
                break
            header_size += 1
        ## now check there is at least 1 line beyond the header:
        status_ok = False
        for (i,line) in enumerate(fid):
            if i > header_size:
                status_ok = True
    if not status_ok:
        return np.array([])

    # Read text: TODO: improve skiprows
    data = np.loadtxt(est_file, skiprows=header_size)
    data = np.atleast_2d(data)
    return data


def get_voicing_mask(ixx, voicing, wavelength):

    changes = (voicing[:-1] - voicing[1:])
    ons = []
    offs = []

    if voicing[0] == 1:
        ons.append(0)

    for (i,change) in enumerate(changes):
        if change < 0:
            ons.append(i)
        elif change > 0:
            offs.append(i+1)

    if voicing[-1] == 1:
        offs.append(len(voicing))

    assert len(ons) == len(offs)

    seq = np.zeros(wavelength)
    for (on, off) in zip(ons, offs):

        on_i = min(ixx[on], wavelength)
        off_i = min(ixx[off-1], wavelength)
        seq[on_i:off_i] = 1.0

    return seq

def synthesise_excitation(fzerofile, wavelength, srate=16000, frameshift_sec=0.005, uv_length_sec=0.005):
    d = read_est_file(fzerofile)
    fz = d[:,2]
    fz_sample = np.repeat(fz, int(srate * frameshift_sec))
    if fz_sample.shape[0] > wavelength:
        fz_sample = fz_sample[:wavelength]
    elif fz_sample.shape[0] < wavelength:
        diff = wavelength - fz_sample.shape[0]
        fz_sample = np.concatenate([fz_sample, np.ones(diff)*fz_sample[-1]])
    
    pm = get_synthetic_pitchmarks(fz_sample, srate, uv_length_sec)
    
    f,b,sawtooth = get_epoch_position_features(pm, srate, wavelength, seconds2samples=False, zero_uv_GCP=False)

    ### TODO: refactor and merge
    fz_at_pm = fz_sample[pm]
    voicing = np.ones(pm.shape)
    voicing[fz_at_pm <= 0.0] = 0

    ## convert to 16bit range for storage later (positives only):
    halfrange = (2**16) / 2
    sawtooth *= halfrange  ## TODO: this conversion reversed a little later! rationalise....
    
    voiced_mask = get_voicing_mask(pm, voicing, wavelength)

    sawtooth = sawtooth.flatten()
    sawtooth *= voiced_mask

    return sawtooth



def get_synthetic_pitchmarks(fz_per_sample, srate, uv_length_sec):  
    '''
    unlike in slm-local stuff, assume F0 is already upsampled, and uv regions are 0 or negative
    '''
    uv_length_samples = uv_length_sec * srate
    ## make pitch marks:
    current = 0
    pms = [current]
    while True:
        val = int(fz_per_sample[current])
        if val <= 0:
            current += uv_length_samples
        else:
            current += srate / val
                
        if current >= len(fz_per_sample):
            break
        
        current = int(current)
        
        pms.append(current)
    return np.array(pms)




def convert_model_for_trace(model):
    # to_store 
    trace_model_in = model.input 
    trace_model_out = []
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
        trace_model_out.append(layer.output)
    trace_model = Model(trace_model_in, trace_model_out)
    return trace_model, layer_names


def main_work():


    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-e', dest='synth_epoch', type=str, required=True) ## str so we can include checkpoints
    a.add_argument('-t', dest='full_trace', action='store_true', default=False)
    a.add_argument('-pm', dest='oracle_pitchmarks', action='store_true', default=False)
    a.add_argument('-o', dest='alternative_synth_dir', default='')
    
    opts = a.parse_args()
    
    config = load_config(opts.config_fname) 

    model_epoch = opts.synth_epoch
    # =========================================

    _, config_name = os.path.split(opts.config_fname)
    config_name = config_name.replace('.cfg','').replace('.conf','')


    if opts.alternative_synth_dir:
        top_synthdir = opts.alternative_synth_dir
    else:
        top_synthdir = get_synth_dir_name(config)  
    synthdir = os.path.join(top_synthdir, 'epoch_%s'%(model_epoch))
    if not os.path.isdir(synthdir):
        os.makedirs(synthdir)

    ## TODO: cp config

    model_dir = get_model_dir_name(config) 
    model_name = os.path.join(model_dir, 'model_epoch_%s'%(model_epoch))    
    assert os.path.isfile(model_name), 'Cannot find model %s'%(model_name) 

    print 'Loading model...'
    try:
        model = load_model(model_name)
    except: # NameError:
        model = config['model']
        model.load_weights(model_name)

    print 'Loaded model:'
    print model
    print model.summary()

    synthesise_from_config(config, model, synthdir, full_trace=opts.full_trace, oracle_pitchmarks=opts.oracle_pitchmarks, dummy_synth=True)

def synthesise_from_config(config, model, synthdir, full_trace=False, oracle_pitchmarks=False, dummy_synth=False):

    '''
    TODO: refactor and pair dummy_synth with model loading 
    '''

    safe_makedir(synthdir)
    
    if full_trace:
        print 'Make model to output all hidden activations'
        trace_model, layer_names = convert_model_for_trace(model)

    wavedir = config['wavedir']

    basenames = get_testset_names(config['test_pattern'], wavedir)

    nsynth = config.get('n_sentences_to_synth', 1)

    if config.get('normalise_spectrogram_in', 'freq') == 'freq_global_norm':
        model_dir = get_model_dir_name(config) ## repeat this to get norm info        
        norm_mean_fname = os.path.join(model_dir, 'spectrogram_mean.npy')    
        norm_std_fname = os.path.join(model_dir, 'spectrogram_std.npy')    
        assert os.path.isfile(norm_mean_fname) and os.path.isfile(norm_std_fname)
        spectrogram_mean = np.load(norm_mean_fname)
        spectrogram_std = np.load(norm_std_fname)
    else:
        spectrogram_mean = None
        spectrogram_std = None

    ### following lines for compatibility with earlier configs (before norm handling rationalised)
    if 'feat_dim' in config:
        input_dimension = config['feat_dim']
    else:
        input_dimension = config['feat_dim_in']

    if 'normalise_melspectrograms' in config:
        normalise_input_features = config['normalise_melspectrograms']
    else:
        normalise_input_features = config.get('normalise_spectrogram_in', 'freq')


    spectrogram_extractor = get_spectrogram_extractor(n_mels=input_dimension, \
            normalise=normalise_input_features, \
            spectrogram_mean=spectrogram_mean, spectrogram_std=spectrogram_std, \
            dft_window=config.get('dft_window', 512), n_hop=config.get('n_hop', 200))

    ## opt_model : waveform preditor chained with spectrogram extractor
    ## model : waveform predictor -- this is the only bit which is saved
    noise_std = config.get('noise_std', 1.0)
    noise_input = config.get('add_noise', False)

    n_hop = config.get('n_hop', 200)

    ## dummy synthesis on loading (because first use of network not optimised)
    # DUMMY_SYNTH = True
    if dummy_synth:
        print 'synthesise dummy audio...'
        wav = exc = np.zeros(n_hop*20).reshape(1,-1)
        (inputs, targets) = tweak_batch((wav, exc), spectrogram_extractor, config, [])  ### []: dummy output transformers 

        combined_prediction = model.predict(x=inputs) 
        print '                           done!'


    i = 0
    for basename in basenames:
        print basename
        wave_fname = os.path.join(wavedir, basename + '.wav')
        outfile = os.path.join(synthdir, basename + '.wav')
        wav, sr = soundfile.read(wave_fname, dtype='int16') ## TODO: check wave read/load @343948

        if oracle_pitchmarks:
            excdir = config['excdir']
            exc_fname = os.path.join(excdir, basename + '.wav')
            exc, sr = soundfile.read(exc_fname, dtype='int16')
        else:
            fzerodir = config['fzerodir']
            f0_fname = os.path.join(fzerodir, basename + '.f0')
            exc = synthesise_excitation(f0_fname, len(wav))

        wav = trim_to_nearest(wav, n_hop).reshape(1,-1)
        exc = trim_to_nearest(exc, n_hop).reshape(1,-1)
        (inputs, targets) = tweak_batch((wav, exc), spectrogram_extractor, config, []) # []: dummy output transformers

        start_time = timeit.default_timer()

        combined_prediction = model.predict(x=inputs) # (1, 37800, 1)

        prediction = combined_prediction.flatten()
        write_wave(prediction, outfile, scale=False)

        spec = inputs[0]
        print ('>>> %s --> took %.2f seconds (%s frames)' % (basename, (timeit.default_timer() - start_time), spec.shape[1]) )

        if full_trace:
            tracefile = outfile.replace('.wav','_trace.hdf')
            f = h5py.File(tracefile, 'w')

            print 'store all hidden activations'
            full_trace = trace_model.predict(x=inputs)
            
            ## write list in order so we can retrieve data in order:
            write_textlist_to_hdf(layer_names, 'layer_names', f)            
            for (output, name) in zip(full_trace, layer_names):
                # if name.startswith('multiply'):
                    assert output.shape[0] == 1 ## single item test batch
                    output = output.squeeze(0) ## remove batch dimension
                    dataset = f.create_dataset(name, output.shape, dtype='f', track_times=False)
                    dataset[:,:] = output
            f.close()
            print 'Wrote %s'%(tracefile)

        i += 1
        if i >= nsynth:
            print 'finished synthesising %s files'%(nsynth)
            break


if __name__ == '__main__':
    main_work()
