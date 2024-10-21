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
Download and uncompress the checkpoints:
- All proteins (79G, uncompressed: 105G): https://kth-my.sharepoint.com/:u:/g/personal/honore_ug_kth_se/ESpoHL-BbeVCs2nfiYVq3X8BqW4jP1TGkRvjkGMK25sizw?e=Obsdoc
- Only CP2C9 protein (419MB, 715MB): https://kth-my.sharepoint.com/:u:/g/personal/honore_ug_kth_se/EYmTeoV2nZtLnbseSxj_HH8BtgbrIHgGGkoK65APjHdehw?e=dhs9En

- Put the .tar.gz in the root directory of the repo and uncompress with e.g.

```bash
tar -xzf lightning_logs_cp2c9.tar.gz
```

The logs are organized as follows: `model_name`/`protein_number`/`fold_number`/`version_number`/`checkpoints`
- `model_name` corresponds to the name of a config file in the `configs/` folder.
- `protein_number` corresponds to the index of a config file in a `configs/model_name` folder.
- `fold_number` corresponds to the fold used for testing the model when the model is trained on DMS data. For MSA trained data, we do not use cross-validation and thus only fold0 is available.
- `version_number` differs from 0 when the same configuration file is run multiple times.
- `checkpoints` contains the model checkpoint for a given model and protein.

Checkpoints can be loaded from the `results/models.ipynb` notebook.
