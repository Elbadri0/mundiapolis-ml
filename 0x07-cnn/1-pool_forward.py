#!/usr/bin/env python3
'''Pooling Forward Prop'''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''Function that performs forward
    propagation over a pooling layer
    of a neural network:'''
    m, imgH, imgw, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    imgH, imgw = (
        imgH-kh
    ) // sh + 1, (
        imgw-kw
    ) // sw + 1
    out = np.zeros(
        (m, imgH, imgw, c)
    )
    for i in range(imgH):
        for j in range(imgw):
            if mode == 'avg':
                out[:, i, j, :] = np.average(
                    A_prev[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        :
                    ],
                    axis=(1, 2)
                )
            if mode == 'max':
                out[:, i, j, :] = np.max(
                    A_prev[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        :
                    ],
                    axis=(1, 2)
                )
    return out
