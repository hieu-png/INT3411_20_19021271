import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

f = open('MFCCresults.txt', 'w')
print("Starting creating DTW results")
for index in range(1, 101):
    print("Currently at file: %d" % (index), end="\r")
    audio_file = './19021271_NguyenTrungHieu/c%d.wav' % (index)
    y, sr = librosa.load(audio_file)

    X = librosa.feature.chroma_cens(y=y, sr=sr)
    noise = np.random.rand(X.shape[0], 200)
    Y = np.concatenate((noise, noise, X, noise), axis=1)
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                   ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
          title='Matching cost function')
    plt.show()
print("Finished creating MFCC results")
f.close()
