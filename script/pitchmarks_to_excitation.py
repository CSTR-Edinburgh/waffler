#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - March 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser

import soundfile as sf

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import numpy as np
    
import pylab
import scipy.interpolate

import warnings 

from segmentaxis import segment_axis

NOISY = False

#### This function taken from MagPhase (https://github.com/CSTR-Edinburgh/magphase)
# check_len_smpls= signal length. If provided, it checks and fixes for some pm out of bounds (REAPER bug)
# fs: Must be provided if check_len_smpls is given
def read_reaper_est_file(est_file, check_len_smpls=-1, fs=-1, usecols=[0,1]):

    # Checking input params:
    if (check_len_smpls > 0) and (fs == -1):
        raise ValueError('If check_len_smpls given, fs must be provided as well.')

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
        return (np.array([]), np.array([]))


    # Read text: TODO: improve skiprows
    m_data = np.loadtxt(est_file, skiprows=header_size, usecols=usecols)
    m_data = np.atleast_2d(m_data)
    v_pm_sec  = m_data[:,0]
    v_voi = m_data[:,1]

    # Protection against REAPER bugs 1:
    vb_correct = np.hstack(( True, np.diff(v_pm_sec) > 0))
    v_pm_sec  = v_pm_sec[vb_correct]
    v_voi = v_voi[vb_correct]

    # Protection against REAPER bugs 2 (maybe it needs a better protection):
    if (check_len_smpls > 0) and ( (v_pm_sec[-1] * fs) >= (check_len_smpls-1) ):
        v_pm_sec  = v_pm_sec[:-1]
        v_voi = v_voi[:-1]
    return v_pm_sec, v_voi

def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-pm', dest='indir', required=True)   
    a.add_argument('-wav', dest='wavedir', required=True)        
    a.add_argument('-o', dest='outdir', required=True, \
                    help= "Put output here: make it if it doesn't exist")
    a.add_argument('-pattern', default='', \
                    help= "If given, only normalise files whose base contains this substring")
    a.add_argument('-ncores', default=1, type=int)
    a.add_argument('-simple', default=False, action='store_true', help='output simple unit pulses')
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.outdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    flist = sorted(glob.glob(opts.indir + '/*.pm'))
    
    executor = ProcessPoolExecutor(max_workers=opts.ncores)
    futures = []
    for pmfile in flist:
        futures.append(executor.submit(
            partial(process, pmfile, opts.wavedir, opts.outdir, pattern=opts.pattern, unit_pulse=opts.simple)))
    return [future.result() for future in tqdm(futures)]



def get_epoch_position_features(pms, rate, nsamples, seconds2samples=True, zero_uv_GCP=False):

    if seconds2samples:
        ## Convert seconds -> waveform sample numbers:-
        pms = np.asarray(np.round(pms * rate), dtype=int)
  
    ## make sure length compatible with the waveform:--
    last = len(pms)-1
    while pms[last] > nsamples:
        last -= 1
    pms = pms[:last]
    if nsamples > pms[-1]:
        pms = np.concatenate([pms, np.array([nsamples])])
    ## addd first 0
    pms = np.concatenate([np.array([0]), pms])
    
    start_end = segment_axis(pms, 2, overlap=1)
    lengths = start_end[:,1] - start_end[:,0]
    
    forwards = []
    backwards = []
    norm_forwards = []
    for length in lengths:
        forward = np.arange(length)
        backward = np.flipud(forward)
        norm_forward = forward / float(length)
        forwards.append( forward )
        backwards.append( backward )
        norm_forwards.append(  norm_forward )
    forwards = np.concatenate(forwards).reshape((nsamples,1))
    backwards = np.concatenate(backwards).reshape((nsamples,1))
    norm_forwards = np.concatenate(norm_forwards).reshape((nsamples,1))

    if zero_uv_GCP:
        #forwards[] = 0.0
        sys.exit('not implemented : zero_uv_GCP')
    return (forwards, backwards, norm_forwards)


def process(pmfile, wavedir, outdir, pattern='', unit_pulse=False):
    _, base = os.path.split(pmfile)
    base = base.replace('.pm','')
    
    if pattern:
        if pattern not in base:
            return

    wavefile = os.path.join(wavedir, base + '.wav')
    wave, fs = sf.read(wavefile)
    # try:
    pm, voicing = read_reaper_est_file(pmfile, check_len_smpls=len(wave), fs=fs)
    if pm.size==0:
        if NOISY:
            print 'Trrouble reading pithcmark file, skip'
        return
    # except:
    #     print 'Could not read %s'%(pmfile)
    #     return 
    if unit_pulse:
        sawtooth = np.zeros(wave.size) ## TODO: poor variable name
        marks = np.asarray(np.round(pm[voicing==1.0] * fs), dtype=int)
        sawtooth[marks] = 0.9999999 ## not 1.0, else become negative at s.astype('int16')
    else:
        f,b,sawtooth = get_epoch_position_features(pm, fs, wave.size)

    v_pm = (pm * fs).astype('int')

    ## convert to 16bit range for storage later (positives only):
    halfrange = (2**16) / 2
    sawtooth *= halfrange
    
    sawtooth = sawtooth.flatten()
    if not unit_pulse:
        interpolator = scipy.interpolate.interp1d(v_pm, voicing, kind='linear', \
                                                    axis=0, bounds_error=False, fill_value='extrapolate')
        voiced_mask = interpolator(np.arange(wave.size))
        #unvoiced_mask = np.ones(voiced_mask.shape) - voiced_mask
        sawtooth *= voiced_mask
    sawtooth = sawtooth.astype('int16')
    
    outfile = os.path.join(outdir, base + '.wav')
    sf.write(outfile, sawtooth, fs)


if __name__=="__main__":

    main_work()

