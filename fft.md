```
Python 3.8.2 (default, Jul 16 2020, 14:00:26)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
>>> from np.fft import _pocketfft_internal as pfi
>>> a,n,axis,norm =  np.exp(2j * np.pi * np.arange(8) / 8),None,-1,None
>>> a = asarray(a)
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=None,
axis=-1,
norm=None
>>> n is None
True
>>> n = a.shape[axis]
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=-1,
norm=None
>>>inv_norm = 1
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=-1,
norm=None,
inv_norm=1
>>> norm is not None
False
>>> norm=="ortho"
False
>>> a, n, axis, is_real, is_forward, inv_norm = a, n, axis, False, True, inv_norm
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=},\n{is_real=},\n{is_forward=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=-1,
norm=None,
inv_norm=1,
is_real=False,
is_forward=True
>>> axis = normalize_axis_index(axis, a.ndim) #関数の詳細見つからず
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=},\n{is_real=},\n{is_forward=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=0,
norm=None,
inv_norm=1,
is_real=False,
is_forward=True
>>> n is None
False
>>> n < 1
False
>>> fct = 1/inv_norm
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=},\n{is_real=},\n{is_forward=},\n{fct=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=0,
norm=None,
inv_norm=1,
is_real=False,
is_forward=True,
fct=1.0
>>> r = pfi.execute(a, is_real, is_forward, fct) #C言語で書かれたライブラリ
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=},\n{is_real=},\n{is_forward=},\n{fct=},\n{r=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=0,
norm=None,
inv_norm=1,
is_real=False,
is_forward=True,
fct=1.0,
r=array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
        2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
        9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
        1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])
>>> output = r
>>> print(f"{a=},\n{n=},\n{axis=},\n{norm=},\n{inv_norm=},\n{is_real=},\n{is_forward=},\n{fct=},\n{r=},\n{output=}")
a=array([ 1.00000000e+00+0.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
        6.12323400e-17+1.00000000e+00j, -7.07106781e-01+7.07106781e-01j,
       -1.00000000e+00+1.22464680e-16j, -7.07106781e-01-7.07106781e-01j,
       -1.83697020e-16-1.00000000e+00j,  7.07106781e-01-7.07106781e-01j]),
n=8,
axis=0,
norm=None,
inv_norm=1,
is_real=False,
is_forward=True,
fct=1.0,
r=array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
        2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
        9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
        1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j]),
output=array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
        2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
        9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
        1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])


#参考までに
>>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
        2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
        9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
        1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])