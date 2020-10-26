import numpy as np
from matplotlib import pyplot as plt
import wave
import struct

A = .1
fs = 44100
f0 = 440
sec = 10


def create_wave(A, f0, fs, t):
    # nポイント
    f1 = 880
    f2 = 1320
    point = np.arange(0, fs * t)
    sin_wave = A * np.sin(2 * np.pi * f0 * point / fs) + A * np.sin(2 *
                                                                    np.pi * f1 * point / fs) + A * np.sin(2 * np.pi * f2 * point / fs)

    sin_wave = [int(x * 32767.0) for x in sin_wave]  # 16bit符号付き整数に変換

    # バイナリ化
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)

    w = wave.Wave_write("440Hz.wav")
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()


create_wave(A, f0, fs, sec)
