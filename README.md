# Interactive Language Learning by Question Answering
--------------------------------------------------------------------------------

Most of the code is written by the original authors: Yuan, Xingdi and Cote, Marc-Alexandre and Fu, Jie and Lin, Zhouhan and Pal, Christopher and Bengio, Yoshua and Trischler, Adam. A part of the Third Year Project's contribution is the integration of a pre-trained DistilBERT model into the agent architecture. The changes can mostly be found in `layers.py`, `agent.py` and `model.py`. The Dockerized version of the code can be accessed through the docker tag of `heidonomm/qait-improved`.

## UPDATED:
Clone repo and cd into it  
If conda not installed on your system, uncomment the first block of code in setup.sh  
Run setup.sh (from zero to training)  
  
NB! `conda activate` sometimes does not want to switch to the new venv. If you're afraid that this may cause a mess with your existing setup, do the first commands until `conda activate` by hand.  

## To install dependencies // Outdated
```
sudo apt update
conda create -p ~/venvs/qait python=3.6
source activate ~/venvs/qait
pip install --upgrade pip
pip install numpy==1.16.4
pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
pip install -U spacy
python -m spacy download en
pip install tqdm h5py visdom pyyaml
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it.


## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## To Train
```
python train.py ./
```

## Citation

Please use the following bibtex entry:
```
@article{yuan2019qait,
  title={Interactive Language Learning by Question Answering},
  author={Yuan, Xingdi and C\^ot\'{e}, Marc-Alexandre and Fu, Jie and Lin, Zhouhan and Pal, Christopher and Bengio, Yoshua and Trischler, Adam},
  booktitle={EMNLP},
  year={2019}
}
```

## License

[MIT](./LICENSE)
