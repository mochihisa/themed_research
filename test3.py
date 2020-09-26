#https://www.nicovideo.jp/watch/sm13283644 niconico ここの周波数を使いました
#https://qiita.com/rild/items/339c5c36f4c1ad8d4325 qiita ここの周波数を使いました

import numpy as np
from matplotlib import pyplot as plt
import wave
import struct
import pyaudio
from scipy.fftpack import fft, ifft
import cv2

#パラメータ
RATE=44100
N=1
CHUNK=1024*N
p=pyaudio.PyAudio()
fn=RATE
nperseg=fn*N

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

fs = RATE#サンプリング周波数
sec = 0.1 #秒
s='aiueo_'
sa='a'
def sin_wav(A,f0,fs,t):
    point = np.arange(0,fs*t)
    sin_wav =A* np.sin(2*np.pi*f0*point/fs)
    return sin_wav

def create_wave(A,f0,fs,t):#A:振幅,f0:基本周波数,fs:サンプリング周波数,再生時間[s]
    sin_wave=0
    print(A[0])
    int_f0=int(f0[0])
    for i in range(0,len(A),1):
        f1=f0[i]
        sw=sin_wav(A[i],f1,fs,t)
        sin_wave += sw
    sin_wave = [int(x * 32767.0) for x in sin_wave]
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write('./aiueo/'+s+'/'+sa+'/'+str(int_f0)+'Hz.wav')
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()

def sound_wave(fu):
    int_f0=int(fu)
    wavfile = './aiueo/'+s+'/'+sa+'/'+str(int_f0)+'Hz.wav'
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    return sig

fig = plt.figure(figsize=(12, 8)) #...1
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
A=(0.07,0.09,0.08,0.19,0.08,0.07) #a
f0=27.5 #261.626
f=(f0,2*f0,3*f0,4*f0,5*f0,6*f0) #a

def B(a0=220):
    B=[]
    r= 1.059463094
    b=1
    for n in range(60):
        b *= r
        B.append(b*a0)
    return B

B=B(27.5)
sk=0

def skf(sk):
    skf=sk
    skf = skf+1
    skf=skf%60
    print(skf)
    f0=B[skf]
    return f0,skf

while True:
    create_wave(A, f, fs, sec)
    sig = sound_wave(f[0])
    #cv2.imshow('image',img)
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(20,20000,(20000-20)/int(fn))
    ax2.set_ylim(0,0.05)
    ax2.set_xlim(20,20000)
    ax2.set_xlabel('Freq[Hz]')
    ax2.set_ylabel('Power')
    ax2.set_xscale('log')
    ax2.plot(2*f*RATE/44100,Pyy)

    #ax2.set_axis_off()
    x = np.linspace(0, 1, sec*nperseg)
    ax1.plot(x,sig)
    ax1.set_ylim(-1,1)
    int_f0=int(f0)
    plt.savefig('./aiueo/'+s+'/'+sa+'/IntensityvsFreq_'+str(int_f0)+'.jpg')
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    img=cv2.imread('./aiueo/'+s+'/'+sa+'/IntensityvsFreq_'+str(int_f0)+'.jpg')
    cv2.imshow('image',img)
    print("f0_{},A_{},B_{},C_{},D_{}".format(int_f0,A[0],A[1],A[2],A[3]))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('e'): # e
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='e'
        #f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0,11*f0)
        A=(0.19,0.09,0.08,0.07,0.08) #qiita
        #A=(0.18,0.14,0.13,0.03,0.03) #niconico
    elif k == ord('c'):
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='e'
        #f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0,11*f0)
        A=(0.19,0.09,0.08,0.07,0.08) #qiita
        #A=(0.18,0.14,0.13,0.03,0.03) #niconico
    elif k == ord('a'): # a
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='a'
        f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0,5*f0,6*f0)
        A=(0.07,0.09,0.08,0.19,0.08,0.07) #qiita
        #A=(0.07,0.09,0.08,0.19,0.08,0.07) #niconico
    elif k == ord('z'):
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='a'
        f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0,5*f0,6*f0)
        A=(0.07,0.09,0.08,0.19,0.08,0.07)  #qiita
        #A=(0.07,0.09,0.08,0.19,0.08,0.07) #niconico
    elif k == ord('i'): # i
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='i'
        #f0,sk=skf(sk)
        f=(f0,2*f0,11*f0,12*f0,13*f0)
        A=(0.19,0.09,0.08,0.07,0.08)  #qiita
        #A=(0.52,0.03,0.02,0.01,0.02) #niconico
    elif k == ord('p'):
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='i'
        #f0,sk=skf(sk)
        f=(f0,2*f0,11*f0,12*f0,13*f0)
        A=(0.19,0.09,0.08,0.07,0.08) #qiita
        #A=(0.52,0.03,0.02,0.01,0.02) #niconico
    elif k == ord('o'): # o
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='o'
        #f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0)
        A=(0.11,0.12,0.12,0.19)  #qiita
        #A=(0.11,0.14,0.10,0.24) #niconico
    elif k == ord('r'):
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='o'
        #f0,sk=skf(sk)
        f=(f0,2*f0,3*f0,4*f0)
        A=(0.11,0.12,0.12,0.19)  #qiita
        #A=(0.11,0.14,0.10,0.24) #niconico
    elif k == ord('u'): # u
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='u'
        #f0,sk=skf(sk)
        f=(f0,2*f0,4*f0,5*f0,6*f0)
        A=(0.19,0.08,0.08,0.08,0.09) #qiita
        #A=(0.32,0.11,0.02,0.02,0.13) #niconico
    elif k == ord('t'):
        cv2.imwrite('./aiueo/'+s+'/'+sa+ '/f0_{}_{}_{}_{}_{}.jpg'.format(int_f0,A[0],A[1],A[2],A[3]),img)
        sa='u'
        #f0,sk=skf(sk)
        f=(f0,2*f0,4*f0,5*f0,6*f0)
        A=(0.19,0.08,0.08,0.08,0.09) #qiita
        #A=(0.32,0.11,0.02,0.02,0.13) #niconico
