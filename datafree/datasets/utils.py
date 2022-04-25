import os
import numpy as np
import math

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def curr_v(l, lamda, spl_type='hard'):
    if spl_type == 'hard':
        v = (l < lamda).float()
        g = -lamda * (v.sum())
    elif spl_type == 'soft':
        v = (l < lamda).float()
        v *= (1 - l / lamda)
        g = 0.5 * lamda * (v * v - 2 * v).sum()
    elif spl_type == 'log':
        v = (1 + math.exp(-lamda)) / (1 + (l - lamda).exp())
        mu = 1 + math.exp(-lamda) - v
        g = (mu * mu.log() + v * v.log() - lamda * v).sum()

    else:
        raise NotImplementedError('Not implemented of spl type {}'.format(spl_type))

    return g, v


def lambda_scheduler(lambda_0, iter, alpha=0.0001, iter_0=500000000):
    if iter < iter_0:
        lamda = lambda_0 + alpha * iter
    else:
        lamda = lambda_0 + alpha * iter_0
    return lamda