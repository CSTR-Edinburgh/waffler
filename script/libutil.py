import os
import re
import numpy as np
import hashlib
import soundfile

def load_config(config_fname):
    config = {}
    execfile(config_fname, config)
    del config['__builtins__']
    _, config_name = os.path.split(config_fname)
    config_name = config_name.replace('.cfg','').replace('.conf','')
    config['config_name'] = config_name
    return config

def safe_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
def writelist(seq, fname):
    f = open(fname, 'w')
    f.write('\n'.join(seq) + '\n')
    f.close()
    
def readlist(fname):
    f = open(fname, 'r')
    data = f.readlines()
    f.close()
    return [line.strip('\n') for line in data]
    
def read_norm_data(fname, stream_names):
    out = {}
    vals = np.loadtxt(fname)
    mean_ix = 0
    for stream in stream_names:
        std_ix = mean_ix + 1
        out[stream] = (vals[mean_ix], vals[std_ix])
        mean_ix += 2
    return out
    
def makedirecs(direcs):
    for direc in direcs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

def basename(fname):
    path, name = os.path.split(fname)
    base = re.sub('\.[^\.]+\Z','',name)
    return base    

get_basename = basename # alias

def get_speech(infile, dimension):
    f = open(infile, 'rb')
    speech = np.fromfile(f, dtype=np.float32)
    f.close()
    assert speech.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    speech = speech.reshape((-1, dimension))
    return speech    

def filehash(fname):
    hasher = hashlib.md5()
    with open(fname, 'rb') as f:
        buffer = f.read()
        hasher.update(buffer)
    return hasher.hexdigest()    

def write_wave(data, fname, rate=16000, scale=True):
    assert fname.endswith('.wav')
    ### do not do data /= data.max() as this will change it in place, and cannot go back if we also want to write unnormed later
    if scale:
        ndata = data / data.max() ## to range [-1, 1]
    else:
        ndata = data
    soundfile.write(fname, ndata, rate)

## from convolin make_data.py
def write_textlist_to_hdf(textlist, name, f):
    textlist = [item.encode('utf8') for item in textlist]  # see https://github.com/h5py/h5py/issues/289
    textarray = np.array(textlist)
    dset = f.create_dataset(name, textarray.shape, dtype=textarray.dtype)
    dset[:] = textarray