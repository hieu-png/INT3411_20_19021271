import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#Reading 100 wav file to extract mel frequency cepstral coefficients
f  = open('MFCCresults.txt','w')
print("Starting creating MFCC results")
for index in range(1,101):
    print("Currently at file: %d" % (index), end="\r")
    audio_file = './19021271_NguyenTrungHieu/c%d.wav'%(index)
    ipd.Audio(audio_file)
    signal, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)

    """ plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs,
                         x_axis="time",
                         sr=sr)
    plt.colorbar(format="%+2.f") """
    #plt.show()

    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    """ plt.figure(figsize=(25, 10))
    librosa.display.specshow(delta_mfccs,
                         x_axis="time",
                         sr=sr)
    plt.colorbar(format="%+2.f") """

    mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    
    #print(mfccs_features.shape)
    f.writelines('%d %d\n' % mfccs_features.shape)
    # plt.show()
print("Finished creating MFCC results")
f.close()