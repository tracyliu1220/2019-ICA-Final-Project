#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image

def one(img):
    return np.ones(img.shape, dtype=int)

def sqr(img):
    return img ** 2

def mul(img, pattern):
    return scipy.signal.fftconvolve(img, np.flip(pattern), mode='valid')
#    return scipy.signal.convolve2d(img, np.flip(pattern), mode='valid')

def sqrDiff(img, pattern):
    mask = np.where(img < 120, 0, 1)
    result = 0
    result += mul(sqr(img) * mask, one(pattern)) 
    result -= 2 * mul(img * mask, pattern)
    result += mul(one(img) * mask, sqr(pattern))
    return result

def toNumber(targets):
    result = 0
    h, w = targets[0].shape
    x, y = 0, 0
    while y < w:
        while y < w and x < h:
            for i in range(10):
                if x >= targets[i].shape[0] or y >= targets[i].shape[1]:
                    continue
                if targets[i][x][y] >= 0:
                    result = 10 * result + i
                    y += 8
                    break
            x = x+1
        y, x = y+1, 0 
    return result

patterns = []
for i in range(10):
    pattern = Image.open('../imgs/numbers/'+str(i)+'.png')
    pattern = np.array(pattern).astype(int)
    patterns.append(pattern)

for j in range(64):
    img = Image.open('../imgs/00{:02}'.format(j+1) + '.png')
    img = np.array(img)[:,:,0].astype(int)
    result = [] 
    for i in range(10):
        d = sqrDiff(img, patterns[i])
        d = np.where(d < 600000, i, -1)
        result.append(d)
    print(j+1, toNumber(result))


