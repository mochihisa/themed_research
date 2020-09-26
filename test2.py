###############################################################################
# 逆高速フーリエ変換（ IFFT ）を計算するプログラム
###############################################################################
# インポート
import matplotlib.pyplot as plt
import numpy as np

# 簡単な信号の作成
N     = 2**6 # サンプル数
dt    = 0.025 # サンプリング周期（ sec ）
freq1 = 10 # 周波数（ Hz ）
ampl1 = 1 # 振幅
freq2 = 15 # 周波数（ Hz ）
ampl2 = 1 # 振幅
print("■■■　入力条件　■■■")
print("サンプル数 : " + str(N))
print("サンプリング周期 : " + str(dt) + ' sec')
print("サンプリング周波数 : " + str(1/dt) + ' Hz')
print("入力波１")
print("　　周波数 : " + str(freq1) + ' Hz')
print("　　振　幅 : " + str(ampl1))
print("入力波２")
print("　　周波数 : " + str(freq2) + ' Hz')
print("　　振　幅 : " + str(ampl2))

# 時間軸
t = np.arange(0, N*dt, dt)
print("サンプリング時間 : " + str(N*dt-dt) + ' sec')
print("サンプリング時刻 : " + str(t))

# ｛周波数　freq1, 振幅　amp1　の正弦入力波｝ + ｛周波数　freq2, 振幅　amp2　の正弦入力波｝
f = ampl1*np.sin(2*np.pi*freq1*t) + ampl2*np.sin(2*np.pi*freq2*t)
# 周波数　freq1, 振幅　amp1　の正弦入力波
#f = amp1 * np.sin(2*np.pi*freq1*t)

# 高速フーリエ変換（ FFT ）
F = np.fft.fft(f)

# FFT の複素数結果を絶対に変換
absf = np.abs(F)

# 振幅をもとの信号に揃える
absf_amp = absf / N * 2
absf_amp[0] = absf_amp[0] / 2

# 周波数軸のデータ作成
fq = np.linspace(0, 1.0/dt, N) # 周波数軸　linspace(開始, 終了, 分割数)

idx = np.argmax(f)
print("\n■■■　入力信号特性　■■■")
print("入力信号の最大idx : " + str(idx))
print("入力信号の最大振幅 : " + str(f[idx]))
print("入力信号の最大時刻 : " + str(t[idx]))

idx = np.array(absf_amp[:int(N/2)+1]) # コピー
idx = idx.argsort()[::-1] # 降順に並べ替えた時のインデックスを取得する
print("\n■■■　フーリエ変換信号特性　■■■")
print("フーリエ変換信号の最大idx : " + str(idx[0]))
print("フーリエ変換信号の最大振幅 : " + str(absf_amp[idx[0]]))
print("フーリエ変換信号の最大周波数 : " + str(fq[idx[0]]))

print("\nフーリエ変換信号の次点idx : " + str(idx[1]))
print("フーリエ変換信号の次点振幅 : " + str(absf_amp[idx[1]]))
print("フーリエ変換信号の次点周波数 : " + str(fq[idx[1]]))

# 逆フーリエ変換（ IFFT ）
F_ifft = np.fft.ifft(F)

# 実数部の取得
F_ifft_real = F_ifft.real

# スライス
F_ifft_real[:10]

idx = np.argmax(F_ifft_real)
print("\n■■■　逆フーリエ変換信号特性　■■■")
print("逆フーリエ変換信号の最大idx : " + str(idx))
print("逆フーリエ変換信号の最大振幅 : " + str(F_ifft_real[idx]))
print("逆フーリエ変換信号の最大時刻 : " + str(t[idx]))

# グラフ表示
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

# 信号のグラフ（時間軸）
axes[0, 0].plot(t, f)
axes[0, 0].set_title('Input Wave')
axes[0, 0].set_xlabel('time[sec]')
axes[0, 0].set_ylabel('amplitude')
axes[0, 0].grid(True)

# FFTのグラフ（周波数軸）
axes[0, 1].plot(fq[:int(N/2)+1], absf_amp[:int(N/2)+1])
axes[0, 1].set_title('Fast Fourier Transform')
axes[0, 1].set_xlabel('freqency[Hz]')
axes[0, 1].set_ylabel('amplitude')
axes[0, 1].grid(True)

# IFFTのグラフ（時間軸）
axes[1, 0].plot(t, F_ifft_real, c="g")
axes[1, 0].set_title('Inverse Fast Fourier Transform')
axes[1, 0].set_xlabel('time[sec]')
axes[1, 0].set_ylabel('amplitude')
axes[1, 0].grid(True)

# Input Wave と IFFT Wave の重ね合わせ
axes[1, 1].plot(t, f, c="g")
axes[1, 1].plot(t, F_ifft_real, c="b")
axes[1, 1].set_title('Input Wave and IFFT')
axes[1, 1].set_xlabel('time[sec]')
axes[1, 1].set_ylabel('amplitude')
axes[1, 1].grid(True)

# グラフの出力
file_dir  = './'
file_name = 'ifft'
fig.savefig(file_dir + file_name + '0.0.jpg', bbox_unches="tight")
