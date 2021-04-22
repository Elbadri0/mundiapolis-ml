#!/usr/bin/env python3
'''Convolutional Forward Props'''
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''that performs forward propagation
    '''
    m, imgh, imgw, c = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride
    imgHP, imgwp = 0, 0
    if padding == 'same':
        imgHP = (
            (
                (imgh - 1) * sh + kh - imgh
            ) // 2
        ) + int(kh % 2 == 0)
        imgwp = (
            (
                (imgw - 1) * sw + kw - imgw
            ) // 2
        ) + int(kw % 2 == 0)
    if type(padding) == tuple:
        imgHP, imgwp = padding
    imgh, imgw = (
        imgh - kh + 2 * imgHP
    ) // sh + 1, (
        imgw - kw + 2 * imgwp
    ) // sw + 1
    out = np.zeros(
        (m, imgh, imgw, knc)
    )
    new = np.pad(
        A_prev,
        (
            (0, 0),
            (imgHP, imgHP),
            (imgwp, imgwp),
            (0, 0)
        ),
        'constant',
        constant_values=0
    )
    for k in range(knc):
        for i in range(imgh):
            for j in range(imgw):
                out[:, i, j, k] = np.sum(
                    new[
                        :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                    ] * W[..., k],
                    axis=(1, 2, 3)
                )
    return activation(out + b)
