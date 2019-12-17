for file in $(ls *.png); do
    mv $file 0$file
done
