
from kapre.time_frequency import Spectrogram, Melspectrogram
from keras.utils.conv_utils import conv_output_length

import numpy as np
from keras import backend as K
from keras import activations

from kapre_modified import amplitude_to_decibel
from pystoi.stoi import utils

def get_octave_band_matrix(fs=10000.0, n_filter=512, n_bands=15, min_freq=150.0):

    octave_band_matrix, CF = utils.thirdoct(float(fs), n_filter, n_bands, min_freq)  # this will be a matrix 15 x 257

    return octave_band_matrix

def get_low_pass_kernel(n_filters, order=2):
    
    # low pass filter FIR (max. flat design) - cutoff freq 1000
    if order == 2:
        b = np.array([0.25, 0.5, 0.25]) 
    else:
        b = np.array([-0.008, 0.064, 0.443, 0.443, 0.064, -0.008])

    # Filtering withing the same channel only...
    order = b.shape[0]
    k = np.zeros( (n_filters , n_filters , order ))
    for n in np.arange(n_filters):
        k[n , n, :] = b

    # Put in expected format
    k = np.transpose(k) # order x n_filters x n_filters
    k = k[: , np.newaxis, :, : ] # order x 1 x n_filters X n_filters  (height x width x in_channels x out_channels)

    return k.astype(K.floatx())

def get_gt_kernels(filterlen, n_channels_per_erb, fs, flow=0.0, fhigh=None):

    # Input settings
    n_channels = int(np.ceil(freqtoaud(fs/2)*n_channels_per_erb))

    if fhigh is None:
        fhigh = float(fs)/2

    # Some definitions
    fc    = erbspace(flow, fhigh, n_channels) # centre frequencies in Hz
    beta  = 1.0183 * audfiltbw(fc) # filter bandwiths in Hz
    delay = 3 / (2 * np.pi * beta )
    scalconst = 2 * np.power( 2 * np.pi * beta , 4 ) / 6 / fs
    nfirst = np.ceil(fs*delay).astype(int)
    nlast  = int(np.floor(filterlen/2))

    # Initializing
    gt_real_kernels = np.zeros((n_channels, filterlen))
    max_size = 0

    # Constructing kernel per freq channel -- padded with zeros -- FIR approximation of Gammatone Filter centered in the ERB scale
    for n in range(n_channels):
        t1 = np.arange(nfirst[n]) / float(fs) - nfirst[n] / float(fs) + delay[n]
        t2 = np.arange(nlast) / float(fs) + delay[n]  
        t  = np.concatenate((t1,t2))
        # The real version
        bwork = scalconst[n] * np.power(t,(4-1)) * np.cos(2*np.pi*fc[n]*t)*np.exp(-2*np.pi*beta[n]*t)
        gt_real_kernels[n,:bwork.shape[0]] = bwork
        max_size = np.max( (max_size,bwork.shape[0]) )

    # Remove unecessary zeros
    gt_real_kernels = gt_real_kernels[:,:max_size] # 34 x 2712

    # Put in expected format
    gt_real_kernels = gt_real_kernels.transpose() # 2712 x 34
    gt_real_kernels = gt_real_kernels[:, np.newaxis, np.newaxis, :] # 2712 x 1 x 1 x 34

    return gt_real_kernels.astype(K.floatx())

def audfiltbw(fc):
    bw = 24.7 + fc/9.265;
    return bw

def erbspace(flow, fhigh, n_channels):
    audlimits = freqtoaud([flow,fhigh])
    y = audtofreq(np.linspace(audlimits[0],audlimits[1],n_channels))
    return y

def audtofreq(aud):
    freq = (1/0.00437)*np.sign(aud)*(np.exp(np.abs(aud)/9.2645)-1)
    return freq

def freqtoaud(freq):
    aud = 9.2645*np.sign(freq)*np.log(1+np.abs(freq)*0.00437)
    return aud

################################################

class SignedMelspectrogram(Melspectrogram):

    def _spectrogram_mono(self, x):

        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')

        # Take magnitude
        output = K.sqrt(output_real ** 2 + output_imag ** 2)

        ## For any alpha
        # alpha = np.pi/2.0
        # phase = K.atan2( output_imag , output_real ) ### this function doesnt exist...
        # a  = K.sign( phase  - ( alpha - np.pi ) )
        # a[a==0] = -1
        # b = K.sign ( alpha - phase )
        # b[b==0] = 1
        # signed_phase = a * b

        ## For alpha = pi/2
        signed_phase = K.sign( output_real )
        output *= signed_phase

        # now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output

class SignedSpectrogram(Spectrogram):

    def _spectrogram_mono(self, x):

        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        # Take magnitude
        output = K.sqrt(output_real ** 2 + output_imag ** 2)

        ## For any alpha
        # alpha = np.pi/2.0
        # phase = K.atan2( output_imag , output_real ) ### this function doesnt exist...
        # a  = K.sign( phase  - ( alpha - np.pi ) )
        # a[a==0] = -1
        # b = K.sign ( alpha - phase )
        # b[b==0] = 1
        # signed_phase = a * b

        ## For alpha = pi/2
        signed_phase = K.sign( output_real )
        output *= signed_phase

        # now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output

class STOIgram(Spectrogram):

    def __init__(self, sr=10000, fmin = 150.0, fmax=None, n_bands=15, trainable_fb=False, **kwargs):

        super(STOIgram, self).__init__(**kwargs)

        assert sr > 0
        assert fmin > 0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert n_bands > 0

        self.sr = int(sr)
        self.fmin = fmin
        self.fmax = fmax
        self.n_bands = n_bands
        self.power_spectrogram = 2.0 # making sure this is set to 2.0 as STOIgram is calculated from the power magnitude
        self.trainable_fb = trainable_fb
        self.n_past = 30

    def build(self, input_shape):
        super(STOIgram, self).build(input_shape)
        self.built = False

        # Compute OB matrix
        octave_band_matrix = get_octave_band_matrix(self.sr, self.n_dft, self.n_bands, self.fmin) # this should be --> (n_freq x n_bands)
        octave_band_matrix = np.transpose(octave_band_matrix)

        self.freq2octave = K.variable(octave_band_matrix, dtype=K.floatx())
        if self.trainable_fb:
            self.trainable_weights.append(self.freq2octave)
        else:
            self.non_trainable_weights.append(self.freq2octave)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_bands, self.n_past, self.n_frame
        else:
            return input_shape[0], self.n_bands, self.n_past, self.n_frame, self.n_ch

    def call(self, x):

        ## Calculate power STFT
        power_spectrogram = super(STOIgram, self).call(x)

        # now,  channels_first: (batch_sample, n_ch, n_freq, n_time)
        #       channels_last: (batch_sample, n_freq, n_time, n_ch)
        if self.image_data_format == 'channels_first':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 1, 3, 2])
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 2, 1])
        # now, whatever image_data_format, (batch_sample, n_ch, n_time, n_freq)

        ## Multiply the power spectogram by the octave band matrix
        output = K.dot(power_spectrogram, self.freq2octave)

        ## Take the square root of it
        output = K.sqrt(output) # shape is (?, 1, ?, 15)  # (batch n_ch n_frame n_bands)  (B C T F)

        ## Get sliding windows
        # To get j sliding windows of N size: [X(m-N+1) .. X(m)] where N=n_past
        num_windows   = K.shape(output)[2] - self.n_past + 1
        start_indices = K.arange(num_windows)
        windows = K.map_fn(lambda t: output[:, :, t:(t + self.n_past), :], start_indices, dtype=K.floatx())
        # windows` is a tensor of shape (num_windows, batch_size, n_past, ...)  (T B C N F)
        windows = K.permute_dimensions(windows, (1, 2, 0, 4, 3)) # B C T F N
        # force n_past dim to be known
        batchsize = K.shape(output)[0]
        num_channels = 1
        windows = K.reshape(windows, shape=(batchsize, num_channels, num_windows, self.n_bands, self.n_past))
        output  = windows

        if self.image_data_format == 'channels_first':
            output = K.permute_dimensions(output, [0, 1, 3, 4, 2])
        else:
            output = K.permute_dimensions(output, [0, 3, 4, 2, 1])

        # print output.shape # (?, 15, 30, ?, 1) OK

        return output

class Auditorygram(Spectrogram):

    def __init__(self, sr=22050, n_filters_per_erb=1, fmin = 0.0, fmax=None, max_size=5000, **kwargs):

        super(Auditorygram, self).__init__(**kwargs)

        assert sr > 0
        assert fmin >= 0.0
        assert n_filters_per_erb > 0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin

        self.sr = int(sr)
        self.n_filters_per_erb = int(n_filters_per_erb)
        self.fmin = fmin
        self.fmax = fmax
        self.max_size = max_size


    def build(self, input_shape):

    	# input_shape is (B x 1 x T)
        self.n_ch    = input_shape[1] # mono / stereo
        self.len_src = input_shape[2]
        self.is_mono = (self.n_ch == 1)
        if self.image_data_format == 'channels_first':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        if self.len_src is not None:
            assert self.len_src >= self.max_size, 'Hey! The input is too short!'

        gt_real_kernels = get_gt_kernels(self.max_size, self.n_filters_per_erb, self.sr, self.fmin, self.fmax) # 34 x 2712 for 1 / 16kHz

        self.n_filters  = gt_real_kernels.shape[3] # 34 
        self.n_dft  = gt_real_kernels.shape[0] # 2712

        self.n_frame = conv_output_length(self.len_src, self.n_dft, self.padding, self.n_hop)
        self.gt_real_kernels = K.variable(gt_real_kernels, dtype=K.floatx(), name="real_kernels")
        
        # kernels shapes: (filter_length, 1, input_dim, nb_filter)?
        if self.trainable_kernel:
            self.trainable_weights.append(self.gt_real_kernels)
        else:
            self.non_trainable_weights.append(self.gt_real_kernels)

        low_pass_kernel = get_low_pass_kernel(self.n_filters)
        self.low_pass_kernel = K.variable(low_pass_kernel , dtype=K.floatx(), name="lp_kernels")

        # Runs the Layer class build -- grandparent...
        super(Auditorygram.__bases__[0], self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_filters, self.n_frame

        else:
            return input_shape[0], self.n_filters, self.n_frame, self.n_ch

    def call(self, x):
        output = self._spectrogram_mono(x[:, 0:1, :])

        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output,
                                        self._spectrogram_mono(x[:, ch_idx:ch_idx + 1, :])),
                                       axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = amplitude_to_decibel(output)

        return output

    def get_config(self):
        config = {'sr': self.sr,
                  'n_filters_per_erb': self.n_filters_per_erb,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'max_size': self.max_size}
        base_config = super(Auditoryogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _spectrogram_mono(self, x):

        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''

        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)

        output = K.conv2d(x, self.gt_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')

        ## Power
        output = K.pow(output,2)

        ## Halfwave rectifier
        # output = activations.relu(output)

        ## Low pass filter -- check! not working..
        output = K.conv2d(output, self.low_pass_kernel, padding=self.padding, data_format='channels_last')

        # Now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output
