#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image

def getPatterns():
    patterns = 255 * np.ones([10, 32, 25], dtype=int)
    for i in range(10):
        pattern = Image.open('../imgs/numbers/' + str(i) + '.png')
        pattern = np.array(pattern).astype(int)
        h, w = pattern.shape
        patterns[i][:h, :w] = pattern
    return patterns
