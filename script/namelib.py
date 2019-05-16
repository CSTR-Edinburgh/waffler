
import os, glob

def get_workdir(config):
    '''
    Default is to put data and trained models under waffler/work/{data,models}. This can be overridden 
    by setting workdir in config file
    '''
    if 'workdir' in config:
        return workdir
    else:
        topdir = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        workdir = os.path.join(topdir, 'work')
        return workdir

def get_model_dir_name(config, override_config_name=''):
    config_name = config['config_name']
    if override_config_name:
        config_name = override_config_name
    return os.path.join(get_workdir(config), 'models', config_name, 'train')

def get_synth_dir_name(config):
    config_name = config['config_name']    
    return os.path.join(get_workdir(config), 'models', config_name, 'synthesis')

def get_data_dir_name(config):
    config_name = config['config_name']
    nfiles = config.get('use_n_files', 0)
    fftlen = config.get('fftlen', 2048)
    m = config['streamdims']['mag']
    p = config['streamdims']['real']    
    return os.path.join(get_workdir(config), 'data', config['dataname'], '%s_%s_%s_%s'%(nfiles, fftlen, m, p))    

def get_hdf_file_name(config):
    chunksize = config['chunksize']
    overlap = config['overlap']
    valid_percent = config['valid_percent']
    prefix = 'data_'
    return os.path.join(get_data_dir_name(config), '%sc%s_o%s_v%s.hdf'%(prefix, chunksize, overlap, valid_percent))
    
def get_testset_names(test_pattern, featdir, ext='.wav'):
    names = [os.path.split(fname)[-1].replace(ext, '') for fname in glob.glob(featdir + '/*' + ext)]    
    names = [fname for fname in names if test_pattern in fname]
    return sorted(names)

def get_trainset_names(test_pattern, featdir, ext='.wav'):
    names = [os.path.split(fname)[-1].replace(ext, '') for fname in glob.glob(featdir + '/*' + ext)]    
    names = [fname for fname in names if test_pattern not in fname]
    return sorted(names)

