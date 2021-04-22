#!/usr/bin/env python3
'''Convolutional Back Prop'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''A Function that performs back
    propagation over a convolutional
    layer of a neural network'''
    m, imgh, imgw, c = dZ.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imgHP, imgwp = 0, 0
    if padding == 'same':
        imgHP = (
            (
                (imgh * sh) - sh + kh - imgh
            ) // 2
        ) + 1
        imgwp = (
            (
                (imgw * sw) - sw + kw - imgw
            ) // 2
        ) + 1
    if type(padding) == tuple:
        imgHP, imgwp = padding
    new = np.pad(
        A_prev, (
            (0, 0),
            (imgHP, imgHP),
            (imgwp, imgwp),
            (0, 0)
        ),
        'constant',
        constant_values=0
    )
    db = np.sum(
        dZ,
        axis=(0, 1, 2),
        keepdims=True
    )
    newDZero = np.zeros(new.shape)
    dW = np.zeros_like(W)
    for n in range(m):
        for i in range(imgh):
            for j in range(imgw):
                for k in range(knc):
                    newDZero[
                        n,
                        i * sh:i * sh+kh,
                        j * sw:j * sw + kw,
                        :
                    ] += np.multiply(
                        dZ[n, i, j, k],
                        W[..., k]
                    )
                    dW[..., k] += np.multiply(
                        dZ[n, i, j, k],
                        new[
                            n,
                            i * sh:i * sh + kh,
                            j * sw:j * sw + kw,
                            :
                        ]
                    )
    if padding == 'same':
        newDZero = newDZero[
            :,
            imgHP:-imgHP,
            imgwp:-imgwp,
            :
        ]
    return newDZero, dW, db
