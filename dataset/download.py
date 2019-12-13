#!/usr/bin/python3

import requests
import shutil
import multiprocessing as mp

def downloadImage(path):
    url = 'https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php'
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

def downloadImages():
    with mp.Pool(processes=64) as pool:
        files = ['{:05}'.format(i) + '.png' for i in range(1000)]
        pool.map(downloadImage, files)
