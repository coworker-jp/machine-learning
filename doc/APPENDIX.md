# Pythonの環境構築

+ Minicondaを推奨
  + Linuxなどだとシステムで使うために最初から入っているPythonがあり、それと環境をわけるため
  + Pythonのバージョン管理やライブラリの管理が楽で、環境構築に失敗してもやり直しやすいため
  + Anacondaだとデフォルトで入っているライブラリが多すぎて容量が大きいため

## 手順
+ [Miniconda](https://docs.conda.io/en/latest/miniconda.html)をインストールする
+ パスを通す
```
$ source ~/.bashrc
```
+ mlという名前の環境を作る(名前は任意につけることができる)
```
$ conda create -n ml
```
+ ml環境を有効にする
```
$ conda activate ml
```
+ pythonをインストールする
```
(ml)$ conda install python==3.7
```
+ pythonのライブラリをインストールする
```
(ml)$ pip install -r requirements.txt
```
