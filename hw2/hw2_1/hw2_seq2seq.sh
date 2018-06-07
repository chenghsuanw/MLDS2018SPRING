#!/bin/bash 
wget -O ./model/model.ckpt.data-00000-of-00001 https://www.dropbox.com/s/qhs0qanpo7awqb7/basic_0.9_512.ckpt.data-00000-of-00001?dl=0
python3 test.py $1 $2