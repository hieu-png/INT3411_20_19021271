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
        





