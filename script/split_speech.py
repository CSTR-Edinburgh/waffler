#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - June 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

    

import librosa
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import soundfile
from libutil import safe_makedir, get_basename
from tqdm import tqdm

# librosa.effects.split(y, top_db=60, ref=<function amax>, frame_length=2048, hop_length=512)[source]

# intervals[i] == (start_i, end_i) are the start and end time (in samples) of non-silent interval i.




def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-o', dest='output_dir', required=True)    
    a.add_argument('-N', dest='nfiles', type=int, default=0)  
    a.add_argument('-l', dest='wavlist_file', default=None)   
    a.add_argument('-ncores', type=int, default=0)   
    opts = a.parse_args()
    
    # ===============================================
    
    split_waves_in_directory(opts.wave_dir, opts.output_dir, num_workers=opts.ncores, tqdm=tqdm, nfiles=opts.nfiles)

def split_waves_in_directory(in_dir, out_dir, num_workers=1, tqdm=lambda x: x, nfiles=0):
    safe_makedir(out_dir)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    wave_files = sorted(glob.glob(in_dir + '/*.wav'))
    if nfiles > 0:
        wave_files = wave_files[:min(nfiles, len(wave_files))]

    for (index, wave_file) in enumerate(wave_files):
        futures.append(executor.submit(
            partial(_process_utterance, wave_file, out_dir)))
    return [future.result() for future in tqdm(futures)]



def _process_utterance(wav_path, out_dir, top_db=30, pad_sec=0.01, minimum_duration_sec=0.5):

    wav, fs = soundfile.read(wav_path)
    pad = int(pad_sec * fs)
    # print pad
    base = get_basename(wav_path)
    # print base
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    starts_ends = librosa.effects.split(wav, top_db=top_db)
    starts_ends[:,0] -= pad
    starts_ends[:,1] += pad
    starts_ends = np.clip(starts_ends, 0, wav.size)
    lengths = starts_ends[:,1] - starts_ends[:,0]
    starts_ends = starts_ends[lengths > fs * minimum_duration_sec]


    for (i, (s,e)) in enumerate(starts_ends):

        ofile = os.path.join(out_dir, base + '_seg%s.wav'%(str(i+1).zfill(4)))
        # print ofile
        soundfile.write(ofile, wav[s:e], fs)

def test():
    safe_makedir('/tmp/splitwaves/')
    _process_utterance('/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/data/nick/wav/herald_030.wav', '/tmp/splitwaves/')

if __name__=="__main__":

    main_work()

