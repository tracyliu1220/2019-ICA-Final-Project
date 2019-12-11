#!/usr/bin/bash

for i in {00000..00099}; do
   curl https://e3new.nctu.edu.tw/theme/dcpc/securimage/securimage_show.php?0.11112242575906661 --output $i.png
   sleep 0.1
done
