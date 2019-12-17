import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def readPatterns():
    targets = []
    for i in range(10):
        targets.append( Image.open('../imgs/numbers/'+str(i)+'.png') )
    return targets

def match(img, targets):
    result = 0
    npimg = np.array(img)

    x, y = 0, 0
    while y < img.width:
        while x < img.height:
            for i in range(10):
                dh, dw = targets[i].height, targets[i].width
                nptarget = np.array(targets[i]).astype(int)
                if x + dh > img.height or y + dw > img.width:
                    continue
                sub = npimg[x:x+dh, y:y+dw].copy().astype(int)
                sub = np.where(sub < 120, targets[i], sub)
                sub -= nptarget
                sub = sub * sub
                score = np.sum(sub)

                if score < 500000:
                    result = 10*result + i
                    y = y+8
                    break
            x = x+1
        y, x = y+1, 0
    return result

if __name__ == '__main__':
    targets = readPatterns()
    for i in range(1,9):
        plt.subplot(3,3,i)
        img = Image.open('../imgs/000{:02}'.format(i) + '.png')
        img = np.array(img)[:,:,0]
        img = Image.fromarray(img)
        imgplot = plt.imshow(img, cmap='gray')
        result = match(img, targets)
        print(result)

    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)

    plt.show()

