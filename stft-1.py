import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys


def main():

    args = sys.argv
    wav_filename = args[1]

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
    ax.set_ylabel('Frequency[kHz]')
    ax.set_xlabel('Time[s]')
    plt.show()


if __name__ == "__main__":
    main()
