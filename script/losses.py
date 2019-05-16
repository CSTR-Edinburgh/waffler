import keras.backend as K

def stoi(y_true, y_pred):
    """ Short term objective intelligibility
    Computes the STOI from one-third octave spectograms
    Expected format of data is: B T F N 1 
    (batch, timestamps, number of freq bands and number of time windows)
    F = 15 and N=30 according to [1]

    Reference 
    [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 
    'A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', 
    ICASSP 2010, Texas, Dallas.
    """

    ## Constants
    axis_windows = 3 # data is in format B T F N 1  / ? ? 15 30 1
    N = y_pred.shape[axis_windows] 
    BETA = -15.0 # Lower SDR bound
    clip_value = 10.0 ** (-BETA / 20.0)
    eps=1e-16

    ## Normalise predicted so that it matches energy of true in each window of 30 frames
    true_energy = K.sum( K.pow(y_true, 2.0) , axis=axis_windows, keepdims=True)
    pred_energy = K.sum( K.pow(y_pred, 2.0) , axis=axis_windows, keepdims=True)
    alpha       = K.sqrt( true_energy / ( pred_energy + eps ) ) 
    y_pred      *= alpha # B T F N 1  / ? ? 15 30 ?

    ## Clip predicted in order to lower bound the signal to distortion ration as described in [1] eq.3
    y_pred = K.minimum( y_pred , y_true * (1 + clip_value) ) # B T F N ?  / ? ? 15 30 ?

    ## Calculate correlation coefficients
    # Subtract the mean vectors
    y_true -= K.mean(y_true, axis=axis_windows, keepdims=True)
    y_pred -= K.mean(y_pred, axis=axis_windows, keepdims=True)
    # Divide by their norms
    y_true /= K.sqrt( K.sum( K.pow(y_true, 2.0), axis=axis_windows, keepdims=True ) ) + eps
    y_pred /= K.sqrt( K.sum( K.pow(y_pred, 2.0), axis=axis_windows, keepdims=True ) ) + eps
    # Get correlation
    correlations_components = y_true * y_pred # B T F N ?
    correlations_components = K.sum( correlations_components , axis=axis_windows) # B T F ?
    correlations_components = K.mean(correlations_components, axis=-1)  # average across channels B T F / ? ? 15

    ## Minimize the negative stoi (same as max stoi)
    correlations_components = -correlations_components

    ## In Keras, the actual optimized objective is the mean of this output array across all dimensions, which will be equivalent to [1] eq.5
    return correlations_components

def spectral_convergence(y_true, y_pred):
    '''
    Inputs should be plain magnitude spectrograms
    '''
    sum_dims = range(1, K.ndim(y_true)) ## dimensions over which to sum (all but batch dimension)
    return K.sqrt ( K.sum( K.square(y_pred - y_true), axis=sum_dims ) )     /      K.sqrt ( K.sum( K.square(y_true), axis=sum_dims ) )


def normalised_mean_squared_error(y_true, y_pred):
    '''
    divide square error in each TF bin by square of reference for that bin
    '''
    eps = 0.00001
    return K.mean((K.square(y_pred - y_true) / (K.square(y_true)+eps)), axis=-1)

