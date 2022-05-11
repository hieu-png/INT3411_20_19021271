import os
import numpy as np
import librosa.display
import IPython.display as ipd
from hmmlearn import hmm
from config import mfcc_file_name


first_line = True
mfcc_data = []
with open(mfcc_file_name) as file:
    for line in file:
        if first_line:
            first_line = False
        else:
            mfcc_data.append(line.split("\t"))


N = 500
D = len(mfcc_data)
states = []
model = hmm.GaussianHMM(n_components=N, covariance_type="full")
model.transmat_ = np.ones((N, N)) / N
model.startprob_ = np.ones(N) / N
fit = model.fit(mfcc_data.T)
z = fit.decode(mfcc_data.T, algorithm='viterbi')[1]
states.append(z)
