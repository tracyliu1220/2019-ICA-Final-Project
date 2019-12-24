# 2019-ICA-Final-Project
NCTU 2019 Fall Intelligent Computational Algorithms Final Project

> 0516009 吳宗達 0611097 曾力揚 0616015 劉姿利

## Data Generation Steps

```
./scripts/download
```
after downloads are done
```
./scripts/label
```

## Run Network

```
cd network
python3 CNN.py
```

## Auto Login

### Installation

```
pip3 install selenium
```
download the [chrome driver](https://chromedriver.chromium.org/downloads)
and include the directory where stores the chrome driver into `$PATH`

### Execution

```
cd demo
python3 autoLogin.py
```

`autoLogin.py` will generate `screenshot.png`
may have to adjust the code `img = img.crop((280, 440, 490, 500))`
to let `screenshot.png` not include any blue area

## Presentation

[slides](http://shorturl.at/ahrt1)
