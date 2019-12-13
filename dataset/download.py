#!/usr/bin/python3

import requests
import shutil
import multiprocessing as mp

def downloadImage(path):
    url = 'https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php'
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

def downloadImages(files):
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(downloadImage, files)

files = ['{:05}'.format(i) + '.png' for i in range(100)]
downloadImages(files)
