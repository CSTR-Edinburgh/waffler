import sys
import os
import glob
import shutil

from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np

from keras.layers import Input
from keras.models import Model, load_model   
import keras.backend as K

from plot_loss import plot_loss
from namelib import get_model_dir_name
from libutil import load_config, write_wave  
import losses
from losses import normalised_mean_squared_error, stoi

from spectrogram_extractor import get_spectrogram_extractor
from data_utils import get_td_warper, tweak_batch
from generate import synthesise_from_config
import transformed_losses
from providers import HDFBatchProvider


def put(string, fname):
    f = open(fname, 'a')
    f.write(string + '\n')
    f.close()


def main_work():


    #################################################
      
    # ======== process command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config_fname', required=True)
    a.add_argument('-X', dest='overwrite_existing_data', action='store_true', \
                    help= "clear any previous training data first")
    a.add_argument('-mail', default='', type=str)
    a.add_argument('-restart', dest='restart_training', default=0, type=int, help='restart from epoch x')

    a.add_argument('-data', dest='hdf_path', default='', type=str) # TODO


    opts = a.parse_args()
    
    assert os.path.isfile(opts.config_fname)

    train_single_configuration(opts.config_fname, opts)


def compile_model(model, config, loss_variable='losses'):

    ### ------ add (derived) losses ------
    main_model_in = model.get_input_at(0) # model.input only works before training has started, later get error:
                            # AttributeError: Layer model_1 has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use `get_input_at(node_index)` instead.    

    main_model_out = model(main_model_in)

    losses = config.get(loss_variable, [])
    assert losses
    output_transformers = []
    transformed_outputs = []
    loss_weights = []
    loss_types = []

    for loss_params in losses:

        if 'type' not in loss_params:
            loss_params['type'] = 'mse'            

        if loss_params['type']=='stoi':
            loss_params['type'] = stoi

        print loss_params
        transformer = transformed_losses.get_transformer(loss_params)
        output_transformers.append(transformer)

        transformed_out = transformer(main_model_out)
        transformed_outputs.append(transformed_out)
        loss_weights.append(loss_params['weight'])
        loss_types.append(loss_params['type'])

    opt_model = Model(main_model_in, transformed_outputs)   
    n = len(transformed_outputs)

    optimiser = config.get('optimiser', 'adam')

    use_normalised_mean_squared_error = config.get('use_normalised_mean_squared_error', False)
    if use_normalised_mean_squared_error:
        opt_model.compile(loss=[normalised_mean_squared_error]*n, loss_weights=loss_weights, optimizer=optimiser)
    else:
        opt_model.compile(loss=loss_types, loss_weights=loss_weights, optimizer=optimiser)


    print opt_model.summary()
    print 'Model compiled with losses: %s'%(losses)
    print 'Weights:'
    print loss_weights
    
    return (opt_model, output_transformers)


def train_single_configuration(config_fname, opts):

    print('Load config from %s'%(config_fname))
    config = load_config(config_fname)    

    fd_loss_type = config.get('fd_loss_type', 'mse')
    assert fd_loss_type in ['spectral_convergence', 'mse']
    if fd_loss_type=='spectral_convergence':
        ### Check output spectrogram extractor is properly configured for use with spectral convergence loss:
        assert config.get('normalise_spectrogram_out', 'freq') == 'none'
        assert config.get('out_spectrogram_type', 'MelSTFT') == 'STFT'
        assert config.get('out_spectrogram_power', 2.0) == 1.0
        assert config.get('decibel_melgram_out', False) == False
        fd_loss_type = losses.spectral_convergence

    hdf_path = config.get('hdf_path', '')
    if not hdf_path:
        if opts.hdf_path:
            hdf_path = opts.hdf_path
        else:
            sys.exit('hdf_path must be specified in config or command line')

    datasets = ['wave', 'excitation']

    take_validation_at_random = config.get('take_validation_at_random', False)
    data_provider = HDFBatchProvider(hdf_path, datasets, limit=config.get('limit_data_percent', 100), valid_percent=config.get('valid_percent', 5), take_validation_at_random=take_validation_at_random)
    batch_size = config['batch_size']
    nepochs = config.get('nepochs', 100)
    n_train_batches = data_provider.get_n_train_batches(batch_size)
    n_valid_batches = data_provider.get_n_valid_batches(batch_size)

    print('train / validation batches: %s , %s'% (n_train_batches, n_valid_batches))
    if not n_train_batches or not n_valid_batches:
        sys.exit('Too few batches')

    if config.get('warped_td_loss', False):
        td_warper = get_td_warper()  ## defined here as needed to process data to get norm stats
    else:
        td_warper = None

    outdir = get_model_dir_name(config)  

    ### Handle the case where we train for X epochs with config 1, copy the model then train more epochs with new config
    if config.get('start_from_trained_config', ''):

        ### TODO: unify this logic with that 20 lines below? Always use fresh directory when restarting?
        if os.path.exists(outdir):
            if opts.overwrite_existing_data:
                shutil.rmtree(outdir)
                os.makedirs(outdir)
            else:
                sys.exit('Directory at %s already exists -- run again with -X to overwrite it'%(outdir))
        else:
            os.makedirs(outdir)

        assert 'restart_training' in config
        old_outdir = get_model_dir_name(config, override_config_name=config['start_from_trained_config'])
        last_model_number = config['restart_training']
        starting_model_fname = old_outdir + '/model_epoch_%s'%(last_model_number) 
        assert os.path.isfile(starting_model_fname), 'File does not exist: %s'%(starting_model_fname)
        copied_starting_model_fname = outdir + '/model_epoch_%s'%(last_model_number) 
        assert not os.path.exists(copied_starting_model_fname)
        shutil.copyfile(starting_model_fname, copied_starting_model_fname)
        ##
        ### store config(s) of previous model for posterity:
        shutil.copyfile(old_outdir + '/config.cfg', outdir + '/config_till_epoch_%s.cfg'%(last_model_number))
        for older_config in glob.glob(old_outdir + '/config_till_epoch_*.cfg'):
             shutil.copy(older_config, outdir)

        for stattype in ['mean', 'std']:
            statfile = old_outdir + '/spectrogram_%s.npy'%(stattype)
            if os.path.exists(statfile):
                shutil.copy(statfile, outdir)     

    if 'restart_training' in config:
        opts.restart_training = config['restart_training']  ## config file overrides command line option
        

    ### ------- get model -------
    start_epoch = 0 # default    
    if os.path.exists(outdir):
        if opts.restart_training:
            last_model_number = opts.restart_training 
            start_epoch = opts.restart_training + 1
            assert last_model_number > 0 ## check not negative
            starting_model_fname = outdir + '/model_epoch_%s'%(last_model_number) 
            assert os.path.isfile(starting_model_fname), 'File does not exist: %s'%(starting_model_fname)
            print('restart at epoch %s by training model %s'%(start_epoch, starting_model_fname))
        else:
            if opts.overwrite_existing_data:
                shutil.rmtree(outdir)
                os.makedirs(outdir)
            else:
                sys.exit('Directory at %s already exists -- run again with -X to overwrite it'%(outdir))
    else:
        os.makedirs(outdir)

    shutil.copyfile(config_fname, outdir + '/config.cfg') ## keep config with model for later reference


    if start_epoch == 0:
        model = config['model']
    else:
        model = load_model(starting_model_fname)  

    with open(outdir + "/history.txt", 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
            
    #model.compile(optimizer=optimiser, loss='mse')

    decibel_melgram_in=config.get('decibel_melgram_in', False)
    decibel_melgram_out=config.get('decibel_melgram_out', False)

    ### TODO: refactor to own function
    if config.get('normalise_spectrogram_in', True) == 'freq_global_norm': ## TODO: only support global norm for input
        print('COLLECTING NORM STATS...')

        norm_mean_fname = os.path.join(outdir, 'spectrogram_mean')    
        norm_std_fname = os.path.join(outdir, 'spectrogram_std') 

        if (os.path.isfile(norm_mean_fname + '.npy') and os.path.isfile(norm_std_fname + '.npy')):  ### might have been copied if restarting
            print ('Input spectrogram norm stat files already exist')
            spectrogram_mean = np.load(norm_mean_fname + '.npy')
            spectrogram_std = np.load(norm_mean_fname + '.npy')            
        else:
            #assert (not os.path.isfile(norm_mean_fname)) and (not os.path.isfile(norm_std_fname))

            ### TODO: refactor to separate function
            spectrogram_extractor_no_norm = get_spectrogram_extractor(n_mels=config['feat_dim_in'], \
                    normalise='none', dft_window=config.get('dft_window', 512), n_hop=config.get('n_hop', 200), \
                    decibel_melgram=decibel_melgram_in, spectrogram_type=config.get('in_spectrogram_type', 'MelSTFT'), \
                    power=config.get('in_spectrogram_power', 2.0))

            spectro_data = []  ## TODO: don't rely on data fitting in memory
            transformers = [] ## placeholder
            for (batch_index, batch) in enumerate(tqdm(range(n_train_batches), desc='Collecting normalisation stats: ')):
                inputs, target = tweak_batch(data_provider.get_train_batch(batch_size), spectrogram_extractor_no_norm, config, transformers) 
                spectro_data.append(inputs[0])

            spectro_data = np.concatenate(spectro_data, axis=0)
            assert K.image_data_format() == 'channels_last'  ## spectrgram extractor returns (batch, time, freq, channel) ; internal format inside normaliser is  ['data_sample', 'freq', 'time', 'channel']
            spectrogram_mean = np.mean(spectro_data, axis=(0,1,3)) # (0,1,3) means average over example, time, final dims, but keep freq dim 2. 
            spectrogram_std = np.std(spectro_data, axis=(0,1,3))

            ### get (1,1,80,1) -> (1,80,1,1) for reasons mentioned a couple of lines above
            spectrogram_mean = np.reshape(spectrogram_mean, (1,-1,1,1))
            spectrogram_std = np.reshape(spectrogram_std, (1,-1,1,1))

            np.save(norm_mean_fname, spectrogram_mean)
            np.save(norm_std_fname, spectrogram_std)

            print('wrote ' + norm_std_fname)
    
    else:
        spectrogram_mean = None
        spectrogram_std = None

    input_spectrogram_extractor = get_spectrogram_extractor(n_mels=config['feat_dim_in'], \
            normalise=config.get('normalise_spectrogram_in', 'freq'), \
            spectrogram_mean=spectrogram_mean, spectrogram_std=spectrogram_std, \
            dft_window=config.get('dft_window', 512), n_hop=config.get('n_hop', 200), \
            decibel_melgram=decibel_melgram_in, spectrogram_type=config.get('in_spectrogram_type', 'MelSTFT'), \
            power=config.get('in_spectrogram_power', 2.0))


    ### compile model: losses are losses to use from epoch 0. Losses called e.g. losses_epoch_3 will be
    ### used from epoch 3 onwards, or until the next epoch for which losses are defined.
    ### In case we are not starting from epoch 0, find the relevant losses to start with:

    counter = start_epoch
    while counter > 0:
        if 'losses_epoch_%s'%(counter) in config:
            loss_variable = 'losses_epoch_%s'%(counter)
            break
        counter -= 1
    if counter == 0:
        loss_variable = 'losses'
    print 'Compile starting model with loss variable: %s'%(loss_variable)
    os.system('sleep 3')
    #if start_epoch==0:
    (opt_model, output_transformers) = compile_model(model, config, loss_variable=loss_variable)


    tune_loss_contributions = config.get('tune_loss_contributions', False)
    if tune_loss_contributions:
        
        assert 'losses' in config, 'losses must be defined when using tune_loss_contributions'
        ### then model will have been compiled
        
        loss_weighter = LossWeighter(initial_variances=[1.0 for thing in transformed_outputs])

        ## get batch of data to work out transformed data sizes:
        example_inputs, example_targets = tweak_batch(data_provider.get_specific_valid_batch(0, batch_size), input_spectrogram_extractor, config, output_transformers) 

        target_outputs = []
        for target in example_targets:
            print target.shape
            target_outputs.append( Input(shape=target.shape[1:]) )  ## [1:] to remove batch dim from shape
        if 0:
            print len(transformed_outputs)
            print len(target_outputs)
            sys.exit('lecnv888')        
        weighted_loss = loss_weighter(transformed_outputs + target_outputs)  ## join lists prediction & reference
        new_inputs = main_model_in + target_outputs  ## flatten to single level list of tensors
        opt_model_loss_tuning = Model(new_inputs, weighted_loss) 
        opt_model_loss_tuning.compile(optimizer=optimiser, loss='mse') 

        print opt_model_loss_tuning.summary()


    ### --- write headers to log file:

    loss_names = ['combined']

    for params in config['losses']:
        name = params['name']    
        i=2     
        while name in loss_names:
            name = name + str(i) 
            i += 1
        loss_names.append(name)

    loss_list = []
    for stage in ['train', 'valid']:
        if tune_loss_contributions and stage=='train':
            loss_list.append('train_combined_loss')
        else:
            for loss in loss_names:
                loss_list.append(stage + '_' + loss)
    loss_list = ' '.join(loss_list)

    put('>> epoch %s'%(loss_list), outdir + "/history.txt")

    saves_per_epoch = config.get('saves_per_epoch', 1)
    if saves_per_epoch > 1:
        save_every = int(n_train_batches / saves_per_epoch)
        save_points = np.arange(1,saves_per_epoch) * save_every
        save_points = dict(zip(save_points, range(len(save_points))))

    if config.get('plot_evolving_pulse', False):
        store_pulse(model, outdir)

    if tune_loss_contributions:
        zero_target = np.zeros((batch_size, 1)) ## because we aim to push combined loss to 0


    ## ----- train loop -----
    for epoch in range(start_epoch, nepochs):

        if epoch > 0 and (not tune_loss_contributions) and 'losses_epoch_%s'%(epoch) in config:
            print 'Compile model with new losses at epoch %s'%(epoch)
            (opt_model, output_transformers) = compile_model(model, config, loss_variable='losses_epoch_%s'%(epoch))



        train_loss = []
        for (batch_index, batch) in enumerate(tqdm(range(n_train_batches), desc='Train (%s, epoch %s): '%(config['config_name'], epoch))):
            inputs, target = tweak_batch(data_provider.get_train_batch(batch_size), input_spectrogram_extractor, config, output_transformers)             
            train_loss.append( opt_model.train_on_batch(x=inputs, y=target) ) # y=np.expand_dims(wav, -1)) )

            if saves_per_epoch > 1:
                if batch_index in save_points:
                    #print 'save checkpoint'
                    cpoint = save_points[batch_index]
                    model.save(outdir + '/model_epoch_%s_%s'%(epoch, str(cpoint).zfill(5)))

        valid_loss = []

        for batch in tqdm(range(n_valid_batches), desc='Validation: '):     
            inputs, target = tweak_batch(data_provider.get_valid_batch(batch_size), input_spectrogram_extractor, config, output_transformers)
                    # opt_type=opt_type, noise_input=noise_input, noise_std=noise_std, predict_residual=predict_residual, fp16=fp16)
            valid_loss.append( opt_model.test_on_batch(x=inputs, y=target) ) #  y=np.expand_dims(wav, -1)) )

        train_losses = np.array(train_loss)
        train_loss = train_losses.mean(axis=0) 
        valid_loss = np.array(valid_loss).mean(axis=0)

        # if opt_type == 'td+fd':
        train_loss = ' '.join([str(element) for element in train_loss])
        valid_loss = ' '.join([str(element) for element in valid_loss])

        modelfile = outdir + '/model_epoch_%s'%(epoch)
        model.save(modelfile)

        optmodelfile = outdir + '/model_epoch_%s.opt'%(epoch)
        opt_model.save(optmodelfile)

        print('Train: %s ; valid: %s'%(train_loss, valid_loss))
        print('Wrote file: %s'%(outdir + '/model_epoch_%s'%(epoch)))

        # -- record loss --
        put('> %s %s %s'%(epoch, train_loss, valid_loss), outdir + "/history.txt")

        # -- record batchwise train losses
        np.save(outdir + "/detailed_history_epoch_%s"%(str(epoch).zfill(4)), train_losses)

        # -- plot loss so far --
        plot_loss(outdir + "/history.txt")

        # --- synthesise 1 batch of validation fragments
        devsynthbatches = [0]
        for dev_batch_num in devsynthbatches:
            inputs, _ = tweak_batch(data_provider.get_specific_valid_batch(dev_batch_num, batch_size), input_spectrogram_extractor, config, output_transformers)                         
            predictions = model.predict(x=inputs) 

            synthesise_valid_batch(predictions, outdir + '/model_epoch_%s_devsynth'%(epoch), \
                        prefix='batch%s-'%(dev_batch_num))
 
        ### -- synthesise first sentences of test data ---- TODO: get valid names?
        synthesise_from_config(config, model, outdir + '/model_epoch_%s_testsynth'%(epoch))


def synthesise_valid_batch(batch_predictions, outdir, prefix=''):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)    

    for (i,fragment) in enumerate(batch_predictions):
        ofile = os.path.join(outdir, prefix + str(i).zfill(3) + '.wav')
        #soundfile.write(ofile, fragment.flatten(), 16000)
        write_wave(fragment.flatten(), ofile)



if __name__=="__main__":
    #rename_model_class('/disk/scratch_ssd/oliver/wp2/train_test07_100/model_epoch_9')
    #test_decay(0.01, 1.1, 100)
    main_work()
    
