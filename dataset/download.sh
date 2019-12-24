#!/usr/bin/bash

if [ $1 = "test" ]; then
    for i in {000000..010000}; do
       curl https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php?0.11112242575906661 --output $i.png
       sleep 0.1
    done
fi

if [ $1 = "train" ]; then
    for i in {000000..200000}; do
       curl https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php?0.11112242575906661 --output $i.png
       sleep 0.1
    done
fi
