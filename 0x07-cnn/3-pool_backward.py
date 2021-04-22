#!/usr/bin/env python3
'''Pooling Back Prop'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''A function that performs back
    propagation over a pooling layer
    of a neural network'''
    m, imgH, imgw, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imghp, imgwp = 0, 0
    newPrev = A_prev
    dA_new = np.zeros_like(A_prev)
    for n in range(m):
        for i in range(imgH):
            for j in range(imgw):
                for k in range(c):
                    if mode == 'max':
                        tmp = newPrev[
                            n,
                            i * sh:i * sh + kh,
                            j * sw:j * sw + kw,
                            k
                        ]
                        mask = tmp == np.max(tmp)
                        dA_new[
                            n,
                            i * sh:i * sh + kh,
                            j * sw:j * sw + kw,
                            k
                        ] += np.multiply(
                            dA[n, i, j, k],
                            mask
                        )
                    if mode == 'avg':
                        dA_new[
                            n,
                            i * sh:i * sh + kh,
                            j * sw:j * sw + kw,
                            k
                        ] += dA[n, i, j, k] / kh / kw
    return dA_new
