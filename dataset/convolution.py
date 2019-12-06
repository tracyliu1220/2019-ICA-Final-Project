#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image

def sqrDiff(img, pattern):
    one = lambda x: np.ones(x.shape, dtype=int)
    sqr = lambda x: x ** 2
    mul = lambda x, y: scipy.signal.fftconvolve(x, np.flip(y), mode='valid')
    
    mask = np.where(img < 120, 0, 1)
    result = 0
    result += mul(sqr(img) * mask, one(pattern)) 
    result -= 2 * mul(img * mask, pattern)
    result += mul(one(img) * mask, sqr(pattern))
    return result

def toNumber(targets):
    arr = -1 * np.ones(200, dtype=int)
    for i in range(10):
        targets[i] = np.amax(targets[i], axis=0)
        L = len(targets[i])
        arr[:L] = np.maximum(arr[:L], targets[i])
    result, i = 0, 0
    while i < 200:
        if arr[i] >= 0:
            result = 10 * result + arr[i]
            i += 8
        i += 1 
    return result

def main():
    patterns = []
    for i in range(10):
        pattern = Image.open('../imgs/numbers/'+str(i)+'.png')
        pattern = np.array(pattern).astype(int)
        patterns.append(pattern)

    for j in range(10000):
        img = Image.open('../imgs/{:05}'.format(j) + '.png')
        img = np.array(img)[:,:,0].astype(int)
        result = [] 
        for i in range(10):
            d = sqrDiff(img, patterns[i])
            d = np.where(d < 600000, i, -1)
            result.append(d)
        print('{:04}'.format(toNumber(result)))

if __name__ == '__main__':
    main()

