from pprint import pprint
import numpy as np
import wave
import struct
from matplotlib import pylab as plt


def text2list(filename):
    res = [line.rstrip('\n') for line in open(filename)]
    res = [s.split('\t') for s in res]
    res.pop(0)
    pprint(res)
    return res


def sin_wave(l):

    A = .1
    fs = 44100
    f0 = 440
    f1 = 880
    f2 = 1320
    t = 10
    sin_wave = 0
    fname = 'sinwave.wav'
    point = np.arange(0, fs * t)
    for s in l:
        A = 100 - abs(int(float(s[1])))
        f = int(float(s[0]))

        if A > 0 :
            A *= 0.000002
            print(A, f)
            sin_wave += A * np.sin(2 * np.pi * f * point / fs)
            #[print(s) for s in sin_wave]
    sin_wave = [int(x * 32767.0) for x in sin_wave]  # 16bit符号付き整数に変換

    # バイナリ化
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)

    w = wave.Wave_write(fname)
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()


def main():
    l = text2list('spectrum.txt')
    sin_wave(l)


if __name__ == "__main__":
    main()
