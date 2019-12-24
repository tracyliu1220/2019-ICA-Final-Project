#!/usr/bin/bash

for i in {000000..$1}; do
   curl https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php?0.11112242575906661 --output $i.png
   sleep 0.1
done
