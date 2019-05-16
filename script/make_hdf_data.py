#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - March 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser

import soundfile
import numpy as np
from segmentaxis import segment_axis

from tqdm import tqdm

from libutil import get_basename

import h5py

def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-wav', dest='wavdir', required=True)    
    a.add_argument('-exc', dest='excdir', required=True)
    a.add_argument('-testpattern', default='')  
    a.add_argument('-trainpattern', default='')         
    a.add_argument('-chunksize', type=int, default=2000)
    a.add_argument('-overlap', type=int, default=100) 
    a.add_argument('-code', action='store_true', default=False)     
    a.add_argument('-o', dest='outdir', required=True) 

    opts = a.parse_args()
    # ===============================================
    
    for direc in [opts.outdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    ### TODO: don't have everything in memory!
    flist = sorted(glob.glob(opts.wavdir + '/*.wav'))
    bases = [get_basename(fname) for fname in flist]

    if opts.code: ## sort by speaker first
        codes = [base.split('_')[-2] for base in bases]
        bases = [base for (code, base) in sorted(zip(codes, bases))]
        speaker_map = sorted(dict(zip(codes, codes)).keys())
        speaker_map = dict(zip(speaker_map, range(len(speaker_map))))


    if opts.testpattern:
        trainbases = [base for base in bases if opts.testpattern not in base]
        bases = trainbases
        print '%s files matching %s held out for testing '%(len(bases) - len(trainbases), opts.testpattern)
    else:
        print 'no test pattern supplied -- no files held out'

    if opts.trainpattern:
        trainbases = [base for base in bases if opts.trainpattern in base]
        bases = trainbases

    bases = [base for base in bases if os.path.isfile(os.path.join(opts.excdir, base + '.wav'))]

    sample_rate = None ## will be set when first wave is opened, and others checked for consistency
    
    condition_name = 'data_c%s_o%s%s.hdf'%(opts.chunksize, opts.overlap)
    outfile = os.path.join(opts.outdir, condition_name)

    f = h5py.File(outfile, 'w')

    todo_list = [(opts.wavdir, 'wave'), (opts.excdir, 'excitation')]

    for (datadir, name) in todo_list:

        wavedata = []
        print 'Reading from %s...'%(datadir)
        for base in tqdm(bases):
            fname = os.path.join(datadir, base + '.wav')
            wave, fs = soundfile.read(fname, dtype='int16') ## TODO: check wave read/load @343948

            if not sample_rate:
                sample_rate = fs
            else:
                assert fs == sample_rate            
            
            wavedata.append(wave)

        print 'concatenate and reshape...'

        wavedata = np.concatenate(wavedata)
        wavedata = segment_axis(wavedata, opts.chunksize, overlap=opts.overlap, end='cut', axis=0)

        print 'Write to HDF...'
        dset = f.create_dataset(name, wavedata.shape, dtype=wavedata.dtype, track_times=False)
        dset[:,:] = wavedata
        print 'Done'
        print 

    if opts.code:
        wavedata = []
        print 'Adding codes...'
        datadir = opts.wavdir
        for base in tqdm(bases):
            fname = os.path.join(datadir, base + '.wav')
            wave, fs = soundfile.read(fname, dtype='int16') ## TODO: check wave read/load @343948

            speaker_id = base.split('_')[-2]
            codes = np.ones(wave.shape, dtype=wave.dtype) * speaker_map[speaker_id]
            wavedata.append(codes)

        print 'concatenate and reshape...'
        wavedata = np.concatenate(wavedata)
        wavedata = segment_axis(wavedata, opts.chunksize, overlap=opts.overlap, end='cut', axis=0)

        print 'Write to HDF...'
        dset = f.create_dataset('speaker_code', wavedata.shape, dtype=wavedata.dtype, track_times=False)
        dset[:,:] = wavedata
        print 'Done'
        print             
        print speaker_map
        print 

    f.close()
    print 'Wrote ' + outfile

    # TODO add file lists


if __name__=="__main__":

    main_work()

