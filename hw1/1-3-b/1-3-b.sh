#!/bin/bash
mkdir model
python3 1-3-b_train.py 8
python3 1-3-b_train.py 64
python3 1-3-b_train.py 256
python3 1-3-b_train.py 1024
python3 1-3-b_train.py 2048
python3 1-3-b_train.py 4096
python3 1-3-b_train.py 8192 
python3 1-3-b_train.py 11264
python3 1-3-b_train.py 16384
python3 1-3-b_train.py 20480






python3 1-3-b.py
