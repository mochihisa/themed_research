import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

# 第1引数：入力ファイル名
# 第2引数：開始位置
# 第3引数：サンプル数
fname = sys.argv[1]
start = int(sys.argv[2])
N = int(sys.argv[3])

# WAVファイル読み込み
# データを0～1に正規化
# データを開始位置からサンプル数だけ取り出し
# 窓関数(numpyのハミング窓)をかける
wavFile = wave.open(fname, 'rb')
channels = wavFile.getnchannels()
amp  = (2**8) ** wavFile.getsampwidth() / 2
fs = wavFile.getframerate()
originalData = wavFile.readframes(-1)
originalData = np.frombuffer(originalData, dtype= "int16") / amp
originalData = originalData[::channels]
originalData = originalData[start:start + N]
window = np.hamming(N)
windowedData = window * originalData
wavFile.close()

# 高速フーリエ変換
# 結果を0~1に正規化
# 周波数軸の値を計算
originalDataSpectrum = np.fft.fft(originalData)
windowedDataSpectrum = np.fft.fft(windowedData)
originalDataSpectrum = originalDataSpectrum / max(abs(originalDataSpectrum))
windowedDataSpectrum = windowedDataSpectrum / max(abs(windowedDataSpectrum))
freqList = np.fft.fftfreq(N, 1 / fs)

# 結果表示
print("分析対象ファイル：" + fname)
print("データ開始位置：", start)
print("サンプル数：", N)
print("チャンネル数：", channels)
print("振幅範囲：", amp)
print("サンプリング周波数：", fs)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title("original data")
plt.plot(originalData)
plt.subplot(2, 2, 2)
plt.title("windowed data")
plt.plot(windowedData)
plt.subplot(2, 2, 3)
plt.plot(freqList, abs(originalDataSpectrum), marker= 'o', linestyle='-')
plt.axis([0, fs / 16, 0, 1])
plt.subplot(2, 2, 4)
plt.plot(freqList, abs(windowedDataSpectrum), marker= 'o', linestyle='-')
plt.axis([0, fs / 16, 0, 1])
plt.show()
