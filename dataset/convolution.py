#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image
import multiprocessing as mp

def sqrDiff(img, pattern):
    one = lambda x: np.ones(x.shape, dtype=int)
    sqr = lambda x: x**2
    mul = lambda x, y: scipy.signal.fftconvolve(x, np.flip(y), mode='valid')

    mask = np.where(img < 120, 0, 1)
    result = 0
    result += mul(sqr(img) * mask, one(pattern))
    result -= 2 * mul(img * mask, pattern)
    result += mul(one(img) * mask, sqr(pattern))
    return result

def toNumber(targets):
    arr = -1 * np.ones(1000, dtype=int)
    for i in range(10):
        targets[i] = np.amax(targets[i], axis=0)
        L = len(targets[i])
        arr[:L] = np.maximum(arr[:L], targets[i])
    result, i = 0, 0
    while i < 1000:
        if arr[i] >= 0:
            result = 10 * result + int(arr[i])
            i += 8
        i += 1
    return result

patterns = []

def readPatterns():
    if len(patterns) > 0:
        return

    for i in range(10):
        pattern = Image.open('../imgs/numbers/' + str(i) + '.png')
        pattern = np.array(pattern).astype(int)
        patterns.append(pattern)

def imgPathToNumber(image_path):
    readPatterns()

    img = Image.open(image_path)
    img = np.array(img)[:, :, 0].astype(int)

    result = []
    for i in range(10):
        d = sqrDiff(img, patterns[i])
        d = np.where(d < 600000, i, -1)
        result.append(d)
    return toNumber(result)

def main():
    # image_paths = ['../imgs/test/{:05}'.format(i) + '.png' for i in range(100)]
    image_paths = ['../imgs/{:06}'.format(i) + '.png' for i in range(170000, 200000)]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(imgPathToNumber, image_paths)
    for x in results:
        print('{:04}'.format(x))

if __name__ == '__main__':
    main()
