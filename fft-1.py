import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


def main():

    args = sys.argv
    wav_filename = args[1]

    rate, data = scipy.io.wavfile.read(wav_filename)
    print(f"{rate=}\n{data=}")
    data = data / 32768

    fft_data = np.abs(np.fft.fft(data))
    freqList = np.fft.fftfreq(data.shape[0], d=1.0 / rate)
    print(f"{fft_data=}\n{freqList=}")

    plt.plot(freqList, fft_data)
    plt.xlim(0, 8000)
    plt.show()


if __name__ == "__main__":
    main()
