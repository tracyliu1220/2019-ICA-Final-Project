from PIL import Image
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def mark(x, y, img, mark=1):
    ret = np.zeros((33, 24))
    for i in range(33):
        for j in range(24):
            ret[i][j] = img[x + i][y + j]
            if mark == 0:
                continue
            if i == 0 or i == 33 - 1:
                img[x + i][y + j] = 0
            if j == 0 or j == 24 - 1:
                img[x + i][y + j] = 0
    ret = Image.fromarray(ret)
    return ret

def trim(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] < 120:
                img[i][j] = 255


def save(img, filename):
    img = img.convert('L')
    img.save('../imgs/' + filename, 'PNG')

def grid(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if i % 20 == 0 or j % 20 == 0:
                img[i][j] = 90
            elif i % 10 == 0 or j % 10 == 0:
                img[i][j] = 200


if __name__ == '__main__':

    im = Image.open('../imgs/00001.png')
    
    # grey mode
    img = np.array(im)      
    img = img[:,:,0] 
    save(Image.fromarray(img), 'test.png')
    
    trim(img)
    # grid(img)
    
    x = 23
    y = 175
    
    num = mark(x, y, img, 1)
    # save(num, 'numbers/4_2.png')
    
    
    im=Image.fromarray(img)
    # im.show()
