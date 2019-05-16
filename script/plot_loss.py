#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

import numpy as np

if 1:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='infile', required=True)
    a.add_argument('-o', dest='outfile', default='', type=str)
    a.add_argument('-clip', dest='clip', default=0.0, type=float, help='max value to plot for loss')
    
    opts = a.parse_args()
    
    # ===============================================


    plot_loss(opts.infile, outfile=opts.outfile, clip=opts.clip)


def plot_loss(infile, outfile='', clip=0.0):

    f = open(infile, 'r')
    data = f.readlines()
    f.close()

    headers = [line.strip(' >\n').split(' ') for line in data if line.startswith('>>')]
    for line in headers:
        assert len(line) == len(headers[0]) 
    header = headers[0]

    data = [line.strip(' >\n').split(' ') for line in data if line.startswith('>') and not line.startswith('>>')]
    data = np.array(data, dtype='f') ## type convert str->float with numpy

    m,n = data.shape
    assert n==len(header), (n, len(header), header)

    if clip > 0.0:
        data = np.clip(data, 0.0, clip)

    #data = np.log(data)
    plt.clf()
    for i in range(1,len(header)):
        plt.plot(data[:,i], label=header[i])
    plt.legend()

    ## Guard against this error, which happens sporadically and unpredictably:
    #  RuntimeError: Failed to open TrueType font
    try:    
        if outfile:
            plt.savefig(outfile)
        else:
            stem = infile.replace('.txt','')
            plt.savefig(stem + '.pdf')
            plt.savefig(stem + '.png')
    except:
        print 'Could not write figures, continue....' 



if __name__=="__main__":

    main_work()

