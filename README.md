<img src="https://github.com/CSTR-Edinburgh/waffler/blob/master/media/waffle_image2.001.png" data-canonical-src="https://github.com/CSTR-Edinburgh/waffler/blob/master/media/waffle_image2.001.png" width="100" height="100" /> 


# waffler 

This repository contains code used to build the proposed systems presented in the following paper:

```
@inproceedings{watts2019speech,
  title={Speech waveform reconstruction using convolutional neural networks with noise and periodic inputs},
  author={Oliver Watts and Cassia Valentini-Botinhao and Simon King},
  booktitle={2019 {IEEE} International Conference on Acoustics, Speech and Signal Processing, {ICASSP} 2019},
  year={2019}
}
```

The instructions below explain how to produce a system comparable to the new system (P0) proposed in that paper.






## Tools

Go to a suitable location and clone repository:

```
git clone https://github.com/CSTR-Edinburgh/waffler.git
cd waffler
WAFFLER=`pwd`
```

We make use of sv56demo for level normalisation - get it like this:


```
mkdir -p $WAFFLER/tool/bin
cd $WAFFLER/tool/
git clone https://github.com/foss-for-synopsys-dwc-arc-processors/G722.git
cd G722/
git checkout 293bd03a21f6ce0adeddf1ef541e0ddc18fea5fc
cd sv56/
make -f makefile.unx
cp sv56demo ../../bin/
```

We make use of Reaper for pitchtracking and GCI detection. Obtain and install like this:

```
cd $WAFFLER/tool/
git clone https://github.com/google/REAPER.git
cd REAPER
mkdir build   # In the REAPER top-level directory
cd build
cmake ..
make
cp $WAFFLER/tool/REAPER/build/reaper $WAFFLER/tool/bin/
```

## Installation of Python dependencies with virtual environment


Make a directory to house virtual environments if you don't already have one, and move to it:

```
mkdir /convenient/location/virtual_python/
cd /convenient/location/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 waffler
source /convenient/location/virtual_python/waffler/bin/activate
```

With the virtual environment activated, you can now install the necessary packages.

```
cd $WAFFLER
pip install --upgrade pip
pip install -r ./requirements.txt  
```



## Arctic voices (ICASSP 2019 experiment)


### Data preparation

The steps below allow you to produce samples comparable to those for system P0 in the paper.

Download the Arctic data (7 speakers):

```
cd $WAFFLER
mkdir database
cd database/

ARCTIC_DB_DIR=http://tts.speech.cs.cmu.edu/awb/cmu_arctic
for VNAME in bdl slt jmk awb rms clb ksp ; do
  wget ${ARCTIC_DB_DIR}/cmu_us_${VNAME}_arctic.tar.bz2 &&
  tar jxvf cmu_us_${VNAME}_arctic.tar.bz2
done
rm *.bz2
```


Split the data into train and test sets (using the same split as the code at `https://github.com/r9y9/wavenet_vocoder`):
```
cd $WAFFLER
python ./experiment_script/make_r9y9_traintest_for_waffler.py ./database/ ./database/
```



Now normalise levels, split (removing most silence), pitchmark the training and test speech:
```
cd $WAFFLER

TRAINDIR=$WAFFLER/database/train/
TESTDIR=$WAFFLER/database/test/

python ./script/normalise_level.py -i $TRAINDIR/wav -o $TRAINDIR/wav_norm -ncores 30
python ./script/split_speech.py -w $TRAINDIR/wav_norm -o $TRAINDIR/wav_norm_split -ncores 30
python ./script/pitchmark_speech.py -i $TRAINDIR/wav_norm_split -o $TRAINDIR/pm -f $TRAINDIR/f0 -ncores 30
python script/pitchmarks_to_excitation.py -pm $TRAINDIR/pm -wav $TRAINDIR/wav_norm_split -o $TRAINDIR/excitation -ncores 30

## test data (no split or excitation generation):

python ./script/normalise_level.py -i $TESTDIR/wav -o $TESTDIR/wav_norm -ncores 30
python ./script/pitchmark_speech.py -i $TESTDIR/wav_norm -o $TESTDIR/pm -f $TESTDIR/f0 -ncores 30
```


Format training data into single file for all speakers:


```
cd $WAFFLER

python script/make_hdf_data.py -wav $TRAINDIR/wav_norm_split -exc $TRAINDIR/excitation -chunksize 16384 -overlap 0 -o ./work/prepared_data/arctic_all
```


And for some individual speakers (this is not necessary to create systems like the one in the ICASSP2019 paper, but lets you experiment with speaker dependent training):

```
for SPKR in awb bdl clb slt ; do
    python script/make_hdf_data.py -wav $TRAINDIR/wav_norm_split -exc $TRAINDIR/excitation -chunksize 16384 -overlap 0 -trainpattern $SPKR -o ./work/prepared_data/arctic_${SPKR}
done    
```



### Training and synthesis (speaker dependent)

Test your setup on a smaller task first (single speaker database for speaker awb).

Modify the first line of the config file `./config/arctic_awb_01.cfg` to point to the full path of your `$WAFFLER` directory. Then:

```
./script/submit_tf.sh ./script/train.py -c ./config/arctic_awb_01.cfg

./script/submit_tf.sh ./script/generate.py -c ./config/arctic_awb_01.cfg -e 1 
```

Synthetic speech is output under `$WAFFLER/work/models/arctic_awb_01/synthesis/epoch_1/*.wav`. Change to value specified by `-e` flag to synthesise from models stored for different epochs of training (starting at 0 for the first epoch). 

Use `./script/submit_cpu.sh` instead of `./script/submit_tf.sh` to run on CPU rather than GPU. The paths hardcoded in these scripts will probably need to be adjusted to suit your environment.

### Training and synthesis (multispeaker)

These commands will build and generate from a system comparable to P0 described in the paper.

Modify the first line of the config file `./config/arctic_multispeaker_01.cfg` to point to the full path of your `$WAFFLER` directory. Then:

```
./script/submit_tf.sh ./script/train.py -c ./config/arctic_multispeaker_01.cfg

./script/submit_tf.sh ./script/generate.py -c ./config/arctic_multispeaker_01.cfg -e 1
```

## Generated speech quality 

Quality of the speech generated by Waffler is still low compared to that produced by conventional vocoders. However, our code allows for fast training and generation, and implementation with Keras allows easy experimentation. We hope that making this repository public will encourage others to experiment and find hyperparameters corresponding to better architectures and optimisation strategies. 

## The name *Waffler*

![Origin of Waffler](https://github.com/CSTR-Edinburgh/waffler/blob/master/media/waffles2.gif "Origin of Waffler")

