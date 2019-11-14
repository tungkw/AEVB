# Variational Auto-Encoder (VAE) 

PyTorch re-implementation of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) by Kingma et al. 2013.

## download dataset
MINST http://yann.lecun.com/exdb/mnist/

FreyFace https://cs.nyu.edu/~roweis/data.html

save to './datasets/'

or specify the arguments



## quick start
```shell
python ./src/VAE_train.py 
--data='FreyFace' \
--data_path='./datasets/FreyFace' \
--batch_size=100 \
--latent_dim=10 \
--hidden_dim=200 \
--learning_rate=0.01 \
--epoch=10000 \
--output_dir='./output'
```

## help
```shell
python ./src/VAE_train.py -h
```
