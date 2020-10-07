import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import scipy.io.wavfile
from multiprocessing import Process


def stft_g(wav_filename):
    wavefile = wave.open(wav_filename, "r")

    nframes = wavefile.getnframes()
    framerate = wavefile.getframerate()

    y = wavefile.readframes(nframes)
    y = np.frombuffer(y, dtype="int16")
    t = np.arange(0, len(y)) / float(framerate)
    print(f"{y=}\n{t=}")

    wavefile.close()

    N = 1024

    freqs, times, Sx = signal.spectrogram(y, fs=framerate, window='hanning',
                                          nperseg=1024, noverlap=N - 100,
                                          detrend=False, scaling='spectrum')
    print(f"{freqs=}\n{times=}\n{Sx=}")
    f, ax = plt.subplots()
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
    plt.show()


def fft_g(wav_filename):
    rate, data = scipy.io.wavfile.read(wav_filename)
    print(f"{rate=}\n{data=}")
    data = data / 32768

    fft_data = np.abs(np.fft.fft(data))
    freqList = np.fft.fftfreq(data.shape[0], d=1.0 / rate)
    print(f"{fft_data=}\n{freqList=}")

    plt.plot(freqList, fft_data)
    plt.xlim(0, 8000)
    plt.show()


def graph(wav_filename):
    rate, data = scipy.io.wavfile.read(wav_filename)
    data = data / 32768
    time = np.arange(0, data.shape[0] / rate, 1 / rate)
    plt.plot(time, data)
    # plt.xlim(0,1000/rate)
    plt.show()


def main():

    args = sys.argv
    wav_filename = args[1]
    p0 = Process(target=fft_g, args=(wav_filename,))
    p0.start()
    p1 = Process(target=stft_g, args=(wav_filename,))
    p1.start()
    p2 = Process(target=graph, args=(wav_filename,))
    p2.start()


if __name__ == "__main__":
    main()
