# matVAE: A Matrix Variational Auto-Encoder for Variant Effect Prediction in Pharmacogenes
Manuscript submitted to ICLR 2025

## Getting started (Ubuntu 22.04)
- Create a python3 environment, activate it, install packages in `requirements.txt` file: 
```bash
virtualenv -p python3 pyenv
ln -s pyenv/bin/activate envpy
. envpy
pip install -r requirements.txt
```

## Run a config file
```bash
python main.py -i configs/matVAEMOG1/1.json -v 0 -j 1
```
