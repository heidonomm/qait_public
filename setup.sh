#!/bin/bash

## if machine does not have conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
# chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
# bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
# conda init bash

## install dependencies
sudo apt update
conda create -p ~/venvs/qait python=3.6
conda activate ~/venvs/qait
pip install --upgrade pip
pip install numpy==1.16.4
pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
pip install -U spacy
python -m spacy download en
pip install gym==0.10.11
pip install tqdm h5py pyyaml visdom scikit-learn matplotlib
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch


## get the testset from ms
wget https://aka.ms/qait-testset
unzip qait-testset


### get the word embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

## convert embeddings to h5
python embedding2h5.py

## train model // runs evaluation at the end, prints out results
python train.py ./