#coding:utf-8
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack
from pylab import *
from levinson_durbin import autocorr, LevinsonDurbin

"""LPCスペクトル包絡を求める"""

def wavread(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    return x, float(fs)

def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

if __name__ == "__main__":
    # 音声をロード
    args = sys.argv
    wav_filename = args[1]
    wav, fs = wavread(wav_filename)
    t = np.arange(0.0, len(wav) / fs, 1/fs)

    # 音声波形の中心部分を切り出す
    center = len(wav) / 2  # 中心のサンプル番号
    cuttime = 0.04         # 切り出す長さ [s]
    s = wav[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]

    # プリエンファシスフィルタをかける
    p = 0.97         # プリエンファシス係数
    s = preEmphasis(s, p)

    # ハミング窓をかける
    hammingWindow = np.hamming(len(s))
    s = s * hammingWindow

    # LPC係数を求める
    lpcOrder = 32
    r = autocorr(s, lpcOrder + 1)
    a, e  = LevinsonDurbin(r, lpcOrder)
    print("*** result ***")
    print("a:", a)
    print("e:", e)

    # LPC係数の振幅スペクトルを求める
    nfft = 2048   # FFTのサンプル数

    fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:int(nfft/2)]

    # オリジナル信号の対数スペクトル
    spec = np.abs(np.fft.fft(s, nfft))
    logspec = 20 * np.log10(spec)
    plot(fscale, logspec[:int(nfft/2)])

    # LPC対数スペクトル
    w, h = scipy.signal.freqz(np.sqrt(e), a, nfft, "whole")
    lpcspec = np.abs(h)
    loglpcspec = 20 * np.log10(lpcspec)
    plot(fscale, loglpcspec[:int(nfft/2)], "r", linewidth=2)

    xlim((0, 10000))
    show()
