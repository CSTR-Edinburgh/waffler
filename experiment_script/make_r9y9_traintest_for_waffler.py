
### rename RY's test set (extracted mel spectrograms) to extended (+speaker) arctic names:


import re, os, sys, glob


arctic_download_location = os.path.realpath(os.path.abspath(sys.argv[1]))
data_output_location=  os.path.realpath(os.path.abspath(sys.argv[2]))

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))

f = open(HERE + '/cmu_testset.txt', 'r')
data = f.readlines()
f.close()


testutts = []
data = [re.split('[-_\t/]', line) for line in data]  # [/_-\t]
for line in data:
  testutts.append(('arctic_' + line[11].replace('.wav',''), line[7]))


flist = glob.glob(arctic_download_location + '/cmu_us_*_arctic/wav/*wav')
print flist


traindir = data_output_location + '/train/wav/'
testdir = data_output_location + '/test/wav/'
os.makedirs(traindir)
os.makedirs(testdir)

for fname in flist:
    fname_parts = fname.split('/')
    base = fname_parts[-1].replace('.wav','')
    spkr = fname_parts[-3].replace('cmu_us_','').replace('_arctic','')
    #print (base, spkr)
    if (base, spkr) in testutts:
        print (base, spkr)
        os.system('ln -s %s %s/%s_%s.wav'%(fname, testdir, base, spkr))
    else:
        os.system('ln -s %s %s/%s_%s.wav'%(fname, traindir, base, spkr))

sys.exit('done!')

