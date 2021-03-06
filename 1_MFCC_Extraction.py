import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from config import mfcc_file_name
#Reading 100 wav file to extract mel frequency cepstral coefficients
f = open(mfcc_file_name, 'w')
#Just writing the header collumn
f.writelines('command\tcoefficients\ttime frames\n')
print('Starting creating MFCC results')
for index in range(1, 101):
    print('Currently at file: %d' % (index), end='\r')
    audio_file_path = './19021271_NguyenTrungHieu/c%d.wav' % (index)
    command_timestamp_path = './19021271_NguyenTrungHieu/c%d.txt' % (index)
    command_array = np.genfromtxt(command_timestamp_path, delimiter='\t', dtype='unicode')
    #print(command_array)
    for command in command_array:
        #print(command[0])

        ipd.Audio(audio_file_path)

        y, sr = librosa.load(
        path=audio_file_path,
        offset= float(command[0]),
        duration=float(command[1])-float(command[0])
        )
        mfccs = librosa.feature.mfcc(y=y, n_mfcc=13, sr=sr)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
        f.writelines('%s\t%d\t%d\n' % (
            command[2], mfccs_features.shape[0], mfccs_features.shape[1]))
print('Finished creating MFCC results')
f.close()
