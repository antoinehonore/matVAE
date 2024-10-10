# matVAE: A Matrix Variational Auto-Encoder for Variant Effect Prediction in Pharmacogenes

## Prerequisite
Download the data folder (2.3G):
- https://kth-my.sharepoint.com/:u:/g/personal/honore_ug_kth_se/EX7yu-YfM8ZApwoS5W_kUzYBevhAS-a-h0rcym7h8zeU5A?e=pNH2ku

- Uncompress (6.4G)

```bash
tar -xzf data.tar.gz
```

## Getting started (Ubuntu 22.04)
- Create a python3 environment, activate it, install packages in `requirements.txt` file: 
```bash
virtualenv -p python3 pyenv
ln -s pyenv/bin/activate envpy
. envpy
pip install -r requirements.txt
```

# Model training

By default, a `lightning_logs` folder is created to store checkpoint and metrics.

## Configuration files
- The configuration files are organized similarly to https://github.com/barketplace/makegrid

## Training a single model
You can train a model using one of the existing config files.
```bash
python main.py -i configs/matVAEMOG1/1.json -v 0 -j 1
```

### Tensorboard
To follow the training on tensorboard:
```bash
tensorboard --logdir=lightning_logs --port 6006
```

Then open http://localhost:6006 in a browser

# Results and checkpoints
Make sure that your virtual environment is installed in jupyter:
```bash
python -m ipykernel install --user --name=pymatVAE
```

Start a jupyter notebook server:
```bash
jupyter notebook
```

## Plotting results
Results can be plotted from `results/results.ipynb` 

## Checkpoints
Checkpoints can be loaded from `results/models.ipynb` 
