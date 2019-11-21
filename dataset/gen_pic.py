from PIL import Image
import sys
import numpy as np
import random
from datetime import datetime

np.set_printoptions(threshold=sys.maxsize)
random.seed(datetime.now())

def img_array(filename):
    im = Image.open(filename)
    img = np.array(im)
    return img

number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

number[4] = img_array('../imgs/numbers/4_1.png')
number[8] = img_array('../imgs/numbers/8_1.png')

def trim_bg(img):
    img = img[:,:,0]       
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 120:
                img[i][j] = 255
    return img

def gen_pic(num, bg):
    # generate numbers
    ret = 255 * np.ones((80, 215))
    x = random.randint(0, 45)
    y = random.randint(0, 90)
    for k in range(4):
        digit = num % 10
        num = num // 10
        for i in range(33):
            for j in range(24):
                _x = x + i
                _y = y + 25 * (3 - k) + j
                ret[_x][_y] = number[digit][i][j]
    # add noise
    bg = trim_bg(bg)
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j] = min(ret[i][j], bg[i][j])

    ret = Image.fromarray(ret)
    return ret

def add_noise(img, bg):
    ret = 255 * np.ones((80, 215))
    bg = trim_bg(bg)
    for i in range(len(img)):
        for j in range(len(img[i])):
            ret[i][j] = min(img[i][j], bg[i][j])
    ret = Image.fromarray(ret)
    return ret


pic = gen_pic(8484, img_array('../imgs/0001.png'))
pic.show()
pic = gen_pic(8484, img_array('../imgs/0002.png'))
pic.show()


