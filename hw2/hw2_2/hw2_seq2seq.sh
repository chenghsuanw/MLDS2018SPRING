#!/bin/bash 
wget -O ./model/model_epoch_final.data-00000-of-00001 https://www.dropbox.com/s/rlrvbu66kc5hkqj/model_epoch_final.data-00000-of-00001?dl=1
python3 test.py $1 $2
