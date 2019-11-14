# Variational Auto-Encoder (VAE) 

PyTorch re-implementation of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma et al. 2013.

## download dataset
MINST http://yann.lecun.com/exdb/mnist/

FreyFace https://cs.nyu.edu/~roweis/data.html

save to 'AEVB/datasets'

or edit ./train.sh 



## quick start
```shell
cd AEVB
bash ./train.sh
```

## setting
```shell
python ./src/AEVB_train.py -h
```