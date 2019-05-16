import h5py
import numpy as np

class HDFBatchProvider(object):
    def __init__(self, hdf_path, datasets, valid_percent=5, take_validation_at_random=False, limit=100):
        '''
        take_validation_at_random: default = False: take the end of the dataset
        '''

        f = h5py.File(hdf_path, 'r')
        self.datasets = [f[dataset] for dataset in datasets]

        if 1:
            print 'Loaded %s'%(hdf_path)
            print 'Datasets:'
            for d in self.datasets:
                print d 
                print d.shape

        length = self.datasets[0].shape[0]

        assert limit in range(1,101)
        if limit < 100:
            limited_length = int(length * 0.01 * limit)
            print 'Reduced %s to %s samples (%s percent)'%(length, limited_length, limit)
            length = limited_length

        assert valid_percent in range(100)
        ntrain = int(length * 0.01 * (100-valid_percent))
         
        assert length > 1, 'need at least 2 training examples to assign validation data'
        if ntrain == length:
            ntrain -= 1  ## minimal validation 

        all_indices = np.arange(length)
        np.random.seed(14355)
        if take_validation_at_random:
            np.random.shuffle(all_indices)

        train_indices = all_indices[:ntrain]
        valid_indices = all_indices[ntrain:length]

        np.random.shuffle(train_indices) 
        self.train_indices = train_indices
        self.valid_indices = valid_indices  ### valid_indices not shuffled at all when take_validation_at_random=False
        
        self.train_start = 0
        self.valid_start = 0

        self.train_length = ntrain
        self.valid_length = length - ntrain


    def get_train_batch(self, batch_size):
        if self.train_start + batch_size > self.train_length:
            self.train_start = 0
        if self.train_start + batch_size > self.train_length:
            sys.exit('Cannot make batch with %s examples from data with only %s examples'%(batch_size, self.train_length))
        ixx = self.train_indices[self.train_start: self.train_start + batch_size]
        ixx = np.sort(ixx)
        batch = [dset[ixx,...] for dset in self.datasets]
        ### TODO: sample importances!
        self.train_start += batch_size
        return batch

    def get_valid_batch(self, batch_size):
        if self.valid_start + batch_size > self.valid_length:
            self.valid_start = 0
        if self.valid_start + batch_size > self.valid_length:
            sys.exit('Cannot make batch with %s examples from data with only %s examples'%(batch_size, self.valid_length))
        ixx = self.valid_indices[self.valid_start: self.valid_start + batch_size]
        ixx = np.sort(ixx)
        batch = [dset[ixx,...] for dset in self.datasets]
        ### TODO: sample importances!
        self.valid_start += batch_size
        return batch

    def get_specific_valid_batch(self, batch_start, batch_size):
        if batch_start + batch_size > self.valid_length:
            sys.exit('Cannot make batch with %s examples from data with only %s examples'%(batch_size, self.valid_length))
        ixx = self.valid_indices[batch_start: batch_start + batch_size]
        ixx = np.sort(ixx)
        batch = [dset[ixx,...] for dset in self.datasets]
        return batch

    def get_n_train_batches(self, batch_size):
        return self.train_length / batch_size

    def get_n_valid_batches(self, batch_size):
        return self.valid_length / batch_size
