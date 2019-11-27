import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def match(img, target):
    npimg = np.array(img)
    nptarget = np.array(target).astype(int)
    arr = []
    dh, dw = target.height, target.width

    for x in range(img.height - dh):
        for y in range(img.width - dw):
            sub = npimg[x:x+dh, y:y+dw].copy().astype(int)
            sub = np.where(sub < 120, target, sub)
            sub -= nptarget
            sub = np.absolute(sub)
            score = np.sum(sub)

            if score < 5000:
                npimg[x][y:y+dw] = 0
                npimg[x+dh-1][y:y+dw] = 0
                arr.append(score)
    img = Image.fromarray(npimg)
    arr.sort()
    print(arr)
    return img

if __name__ == '__main__':
    plt.subplot(8,8,64)
    target = Image.open('../imgs/numbers/9.png')
    imgplot = plt.imshow(target, cmap='gray')

    for i in range(1,64):
        plt.subplot(8,8,i)
        img = Image.open('../imgs/00{:02}'.format(i) + '.png')
        img = np.array(img)[:,:,0]
        img = Image.fromarray(img)
        img = match(img, target)
        imgplot = plt.imshow(img, cmap='gray')

    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)

    plt.show()

