#!/bin/bash
mkdir model
python3 train_hess.py
python3 b2.py

python3 1-2-b_train.py
python3 b3_plot.py
