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
    fname = 'sinwave.wav'
    ch = 1
    width = 2
    fs = 44100
    time = 3
    samples = time * fs
    s = 0
    t = np.linspace(0, time, samples + 1)   
    count = 0
    for list in l:
        if float(list[1]) > -100:
            print(
                f'A={(100 - abs(int(float(list[1]))))},f={int(float(list[0]))}')
            s1 = (100 - abs(int(float(list[1])))) / 35 * \
                np.sin(2 * np.pi * int(float(list[0])) * t)
            print(s1)
            s = s + s1

    print(s)

    l = [[58, 216], [59, 431], [66, 635], [73, 855], [73, 1039], [73, 1282]]
    for ll in l:
        s1 = ll[0] * np.sin(2 * np.pi * ll[1] * t)
        s = s + s1
    s = np.rint(s)
    s = s.astype(np.int16)
    s = s[0:samples]
    data = struct.pack("h" * samples, *s)
    wf = wave.open(fname, 'w')
    wf.setnchannels(ch)
    wf.setsampwidth(width)
    wf.setframerate(fs)
    wf.writeframes(data)
    wf.close()

    plt.plot(s[0:500])
    plt.show()


def main():
    l = text2list('spectrum.txt')
    sin_wave(l)


if __name__ == "__main__":
    main()
