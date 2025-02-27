import os, sys
import argparse
import pandas as pd
import json
import hashlib
import socket
from functools import partial
from parse import parse 
from glob import glob 
from functools import partial
from datetime import datetime
import matplotlib
import socket
import matplotlib.pyplot as plt
if socket.gethostname()=="cmm0958":
    matplotlib.use('tkagg') 
else:
    matplotlib.use('agg') 


from multiprocessing.dummy import Pool
import numpy as np

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader, ConcatDataset

from utils.data import prepare_dms_dataloaders, prepare_msa_dataloaders, get_num_workers,get_train_val_idxes
from utils.data import TheDataset, TheDatasetDMS, CustomBatchSampler, Subset
from utils.models import dmsEVE#minEVE, 
from utils.trainer import lightningdmsEVE#lightningEVE,
from utils import data as data_utils

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.utilities import CombinedLoader    
from sklearn.model_selection import train_test_split
from utils_tbox.utils_tbox import read_pklz, write_pklz
import warnings
warnings.filterwarnings("ignore", message=".*initialize NVML.*")

import lightning.pytorch as pl

#### LOGGING
import logging
# Configure logging at the root level of Lightning
logging.getLogger("lightning").setLevel(logging.ERROR)

def get_input_size(L, d, model_parameters):
    model_parameters["L"] = L
    model_parameters["d"] = d

    model_parameters["vae_input_size"] = L * d
    
    if "Lmax" in  model_parameters["model_type"]:
        #model_parameters["variable_length_H"] = min([L,model_parameters["variable_length_H"]])
        model_parameters["vae_input_size"] = model_parameters["variable_length_H"] * d #n = model_parameters["variable_length_H"]
        
    if "mean" in  model_parameters["model_type"]:
        model_parameters["vae_input_size"] = 1 * d

    return model_parameters

def prot_name(s):
    splitted = os.path.basename(s).split("_")
    if len(splitted)>3:
        return "_".join(splitted[:2])
    else:
        return splitted[0]

def create_mapfiles(MSA_data_folder):
    exclude= ["ancsz_b0.4.a2m"]

    all_msa_files = glob(os.path.join(MSA_data_folder, "*.a2m"))
    all_msa_names = list(map(prot_name, all_msa_files))
    all_msa_files_dict = {k: [fname for fname in all_msa_files if k in fname] for k in all_msa_names}
    all_msa_files_dict = {k: v if len(v)==1 else v[:1] for k,v in all_msa_files_dict.items()}
    assert(all([len(v)==1 for v in all_msa_files_dict.values()]))
    df_msa = pd.DataFrame()
    df_msa["protein_name"] = [k for k,v in all_msa_files_dict.items()]
    df_msa["msa_location"] = [os.path.basename(v[0]) for v in all_msa_files_dict.values()]
    df_msa["theta"] = 0.1

    # all
    df_msa.to_csv("data/mappings/all.csv", index=False,sep=",")
    F = []
    # individual
    for i in range(df_msa.shape[0]):
        fname = "{}.csv".format(df_msa.loc[i]["protein_name"])
        df_msa[i:i+1].to_csv("data/mappings/{}".format(fname), index=False, sep=",")
        F.append("\""+fname+"\"")

    # humans
    df_msa[df_msa["protein_name"].apply(lambda s: "HUMAN" in s)].to_csv("data/mappings/human.csv", index=False, sep=",")

    return df_msa

def get_data(protein_index, mapping_file=None, args=None, training_parameters=None, data_parameters=None, DMS_data_location=None, device=None,seed=None,num_workers=None,pin_memory=False,DMS_ref_df=None):
    protein_name = mapping_file['protein_name'][protein_index]
    protein_ref_rows = DMS_ref_df[DMS_ref_df["UniProt_ID"] == protein_name][["MSA_filename","MSA_theta", "weight_file_name","target_seq"]].drop_duplicates()
    all_dms_files = [os.path.join(DMS_data_location,fname) for fname in DMS_ref_df[DMS_ref_df["UniProt_ID"] == protein_name]["DMS_filename"].values.tolist()]
    
    if protein_ref_rows.shape[0] > 1:
        print("Multiple references for {} ({})\nKeeping last...".format(protein_name,protein_ref_rows.shape[0]))
        protein_ref_rows = protein_ref_rows.loc[[protein_ref_rows.index[-1]]]

    msa_filename, theta, weights_filename, target_seq = protein_ref_rows.values.reshape(-1)

    msa_location = args.MSA_data_folder + os.sep + msa_filename  ###mapping_file['msa_location'][protein_index]
    weight_location = args.MSA_weights_location + os.sep + weights_filename
    
    data_fname = os.path.join("data", "preprocessed", "{}.pklz".format(protein_name))
    #data_fname = os.path.join("/mnt/berzelius/EVE/data", "preprocessednew", "{}.pklz".format(protein_name))
    mse_nlim = 1e3 if socket.gethostname() == "cmm09588" else 1e9

    if not os.path.isfile(data_fname):
        print("MSA file: " + str(msa_location))

        MSA_dataset, DMS_datasets = data_utils.get_protein_data(
            protein_name, msa_location, weight_location, theta, data_parameters, training_parameters, all_dms_files, 
                device=device, seed=seed, num_workers=num_workers,pin_memory=pin_memory,target_seq=target_seq,mse_nlim=mse_nlim
        )
        write_pklz(data_fname, [MSA_dataset, DMS_datasets])
    else:
        print("Reading",data_fname)
        MSA_dataset, DMS_datasets = read_pklz(data_fname)

    return MSA_dataset, DMS_datasets

def get_profilter(profiler):
    if not (profiler is None): 
        #num_training_steps = 20
        from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
        if profiler=="simple":
            profiler = SimpleProfiler(filename="{}_profiler_results.txt".format(profiler))

        elif profiler=="advanced":
            profiler = AdvancedProfiler(filename="{}_profiler_results.txt".format(profiler))
    return profiler#, num_training_steps

# Find the last checkpoint
def get_last_checkpoint(retrain,log_dir,exp_name):
    latest_checkpoint = None
    if not args.retrain:
        earlier_runs = glob(os.path.join(log_dir, exp_name, "version*"))
        if len(earlier_runs)>0:
            latest_run = sorted(earlier_runs, key= lambda s: parse("version_{:d}", os.path.basename(s))[0])[-1]
            earlier_runs_checkpoints = glob(os.path.join(latest_run,"checkpoints","*.ckpt"))
            if len(earlier_runs_checkpoints) > 0:
                latest_checkpoint = earlier_runs_checkpoints[0]
    return latest_checkpoint


def combine_loaders(data_parameters,training_parameters,train_sequential_loader,train_DMS_prot_dataloaders,test_MSA_prot_loaders,val_MSA_prot_loaders,test_DMS_prot_dataloaders,pin_memory,persistent_workers,num_workers):
    train_dataloaders = {}
    n_steps_per_epochs = 0
    if "msa" in data_parameters["training_data"]:
        train_dataloaders["msa"] = train_sequential_loader
        n_steps_per_epochs += len(train_dataloaders)

    if "dms" in data_parameters["training_data"]:
        # Create a combined dataloader of MSA and DMS data
        assert( (training_parameters["validation_set_pct"] > 0) and (training_parameters["validation_set_pct"]<1)),\
        "Need a proper split between training and validation if DMS data is used in training"
        concat_dms_datasets = ConcatDataset([loader.dataset for loader in train_DMS_prot_dataloaders.values()])
        train_dataloaders = CombinedLoader({**train_dataloaders, 
                                            "dms": DataLoader(concat_dms_datasets, batch_size=training_parameters['batch_size'], 
                                                                num_workers=get_num_workers(len(concat_dms_datasets), training_parameters['batch_size'], max_workers=num_workers), 
                                                                shuffle=True, pin_memory=pin_memory,persistent_workers=persistent_workers)
                                            }, mode="max_size_cycle")
        n_steps_per_epochs += len(iter(train_dataloaders))
    
    ## Aggregate the dataloaders of the train and test prot
    val_loaders = {**test_MSA_prot_loaders, **test_DMS_prot_dataloaders}
    if "msa" in data_parameters["training_data"]:
        val_loaders = {**val_loaders,**val_MSA_prot_loaders}
    combined_val_loaders = CombinedLoader(val_loaders, mode="sequential")
    return train_dataloaders, val_loaders, combined_val_loaders,n_steps_per_epochs

def main(args):

    ##################################################################
    ####             Configuration
    ##################################################################
    # Read config file
    model_params = json.load(open(args.model_parameters_location))

    training_parameters = model_params["training_parameters"]
    validation_set_pct = training_parameters["validation_set_pct"]
    num_training_steps = training_parameters["num_training_steps"]

    data_parameters = model_params["data_parameters"]

    verbose = args.v
    log_dir = "lightning_logs"

    # Computation parameters: n_jobs, gpu ...
    num_threads = torch.get_num_threads()
    device = "cpu" if torch.cuda.is_available() else "cpu"
    
    n_jobs = args.j
    num_workers = n_jobs #if torch.cuda.is_available() else num_threads
    AVAIL_devices = 1 if torch.cuda.is_available() else max([n_jobs,1])

    pin_memory = not args.nopin if torch.cuda.is_available() else False
    persistent_workers = args.persist and (num_workers>0)

    # Exp_name_stem
    exp_name_ = os.path.join(os.path.basename(os.path.dirname(args.model_parameters_location)), 
                             os.path.basename(args.model_parameters_location).replace(".json",""))
    
    # Read the list of proteins
    fname = data_parameters["msa_list"]
    MSA_list = os.path.join(".", "data/mappings", fname)
    mapping_file = pd.read_csv(MSA_list)
    n_proteins = len(mapping_file)

    # Protein level training and test sets
    train_prots = np.ones(mapping_file['protein_name'].shape[0],dtype=bool)
    train_prots[data_parameters["testprot"]] = False

    training_proteins = mapping_file['protein_name'].iloc[train_prots]
    test_proteins = mapping_file['protein_name'].iloc[~train_prots]

    ##################################################################
    ####             DATA: Read MSA and DMS
    ##################################################################
    val_loaders = {}

    DMS_ref_file = "data/DMS_substitutions.csv"; 
    DMS_ref_df = pd.read_csv(DMS_ref_file)
    DMS_data_location = args.DMS_data_folder
    
    func = partial(get_data, mapping_file=mapping_file, args=args, data_parameters=data_parameters, training_parameters=training_parameters, DMS_ref_df=DMS_ref_df,
                             DMS_data_location=DMS_data_location, device=device, seed=args.seed, num_workers=num_workers, pin_memory=pin_memory)
    if (n_jobs < 2) or (n_proteins<2):
        out = list(map(func, range(n_proteins)))
    else:
        with Pool(n_jobs) as pool:
            out = pool.map(func, range(n_proteins))
    print("DATA LOADED")
    
    random_state =None#12345
    #torch.manual_seed(random_state)
    
    pl.seed_everything(args.seed, workers=True)
    
    ##################################################################
    ####             DATA: Split train, val, test
    ##################################################################
    ## MSA
    all_MSA_datasets = {k: v[0] for k,v in zip(mapping_file["protein_name"], out)}
    _, train_sequential_loader, val_MSA_prot_loaders, test_MSA_prot_loaders =\
        prepare_msa_dataloaders(all_MSA_datasets, training_proteins, test_proteins, num_workers, pin_memory, model_params,validation_set_pct=validation_set_pct,persistent_workers=persistent_workers,random_state=random_state)

    # Infer L and d
    max_length = max([dataset.x.shape[1] for dataset in all_MSA_datasets.values()])
    embed_dim = max([dataset.x.shape[2] for dataset in all_MSA_datasets.values()])   

    ## DMS
    all_DMS_datasets = {k: v[1] for k,v in zip(mapping_file["protein_name"], out)} 
    
    # Write protein name in DMS datasets objects
    for k, v in all_DMS_datasets.items():
        for kk, vv in v.items():
            vv.protein_name = k

    assert max_length == max([max_length, max([d.x.shape[1] for v in all_DMS_datasets.values() for d in v.values()])]), "Mismatch between MSA and DMS seq_len"

    train_DMS_prot_datasets_all = {k: all_DMS_datasets[k] for k in training_proteins}
    test_DMS_prot_datasets_all =  {k: all_DMS_datasets[k] for k in test_proteins}
    
    assert len(train_DMS_prot_datasets_all)==1, "Training with multiple proteins is NYI, because of multiple DMS per proteins"
    theproteinname = list(train_DMS_prot_datasets_all.keys())[0]
    
    use_dms_factor = model_params["training_parameters"]["use_dms_factor"]

    model_params["model_parameters"]["latent_to_dms_output_size"] = model_params["training_parameters"]["latent_to_dms_target"].count('+')+1

    # By default, make one training per DMS dataset, unless the training data is only MSA
    one_training_per_dms_dataset = data_parameters["training_data"] != "msa"
    for dms_dataset_name, thedmsdataset in train_DMS_prot_datasets_all[theproteinname].items():
        if one_training_per_dms_dataset:
            train_DMS_prot_datasets = {theproteinname: {dms_dataset_name: thedmsdataset}}
            
            test_DMS_prot_datasets = {}

            if len(test_DMS_prot_datasets_all) > 0:
                test_DMS_prot_datasets = {theproteinname: {dms_dataset_name: test_DMS_prot_datasets_all[theproteinname][dms_dataset_name]}}
        
        else:
            # By pass the loop
            train_DMS_prot_datasets = train_DMS_prot_datasets_all
            test_DMS_prot_datasets = test_DMS_prot_datasets_all
            dms_dataset_name = "all_dms_datasets"

        ##################################################################
        ####             DATA: Split validation
        ##################################################################
        # Compute the folds
        if "dms" in data_parameters["training_data"]:
            from sklearn.model_selection import KFold
            n_folds = 5  #int(1 / validation_set_pct)
            train_val_idxes = {protein_name: {dataset_name: list(KFold(n_folds, shuffle=True, random_state=random_state).split(np.arange(len(v[dataset_name]))))
                                            for dataset_name in v.keys()
                                            }
                            for protein_name, v in train_DMS_prot_datasets.items()
                            }
        else:
            n_folds = 1
            train_val_idxes = {protein_name: {dataset_name: [get_train_val_idxes(len(v[dataset_name]), 1, random_state=random_state)]
                                            for dataset_name in v.keys()
                                            }
                            for protein_name, v in train_DMS_prot_datasets.items()
                            }

        all_folds_results = []

        # Iterate over the folds
        for fold_idx in range(n_folds):
            exp_name = os.path.join(exp_name_, os.path.basename(dms_dataset_name), "fold{}".format(fold_idx))
            outputfname = os.path.join(log_dir, os.path.dirname(exp_name), "results.pklz.fold{}".format(fold_idx))
            
            print(exp_name)
            num_training_steps = training_parameters["num_training_steps"]

            if os.path.exists(outputfname) and (not args.retrain):
                results = read_pklz(outputfname)

            else:
                ##################################################################
                ####             DATA: Combine 
                ##################################################################

                train_DMS_prot_dataloaders, test_DMS_prot_dataloaders = \
                    prepare_dms_dataloaders(train_DMS_prot_datasets, test_DMS_prot_datasets, num_workers, pin_memory, training_parameters['batch_size'],
                    fold_idx=fold_idx, train_val_idxes=train_val_idxes, verbose=verbose, persistent_workers=persistent_workers)

                train_dataloaders, val_loaders, combined_val_loaders,n_steps_per_epochs = combine_loaders(
                        data_parameters,training_parameters,
                        train_sequential_loader,train_DMS_prot_dataloaders,test_MSA_prot_loaders,val_MSA_prot_loaders,test_DMS_prot_dataloaders,
                        pin_memory, persistent_workers, num_workers
                )

                if ("msa" in model_params["data_parameters"]["training_data"]) and ("dms" in model_params["data_parameters"]["training_data"]):
                    n_training_msa_samples = len(train_sequential_loader.dataset)
                    n_training_dms_samples = sum([len(loader.dataset) for loader in train_DMS_prot_dataloaders.values()])
                    #  model_params["training_parameters"]["use_dms_factor"] = min([1,use_dms_factor*(n_training_dms_samples/n_training_msa_samples)])
                    model_params["training_parameters"]["use_dms_factor"] = min([use_dms_factor,use_dms_factor*(n_training_dms_samples/n_training_msa_samples)])

                ##################################################################
                ####             MODEL
                ##################################################################
                if n_proteins == 1:
                    T = all_MSA_datasets[list(all_MSA_datasets.keys())[0]].x.shape[1]
                    model_params["model_parameters"]["variable_length_Lmax"] = T

                model_params["model_parameters"] = get_input_size(max_length, embed_dim, model_params["model_parameters"])
                
                torch.manual_seed(args.seed)

                latest_checkpoint = get_last_checkpoint(args.retrain, log_dir, exp_name)

                if "finetune" in training_parameters.keys():
                    if training_parameters["finetune"] != "none":
                        base_model = training_parameters["finetune"]
                        map_location=None
                        if socket.gethostname() == "cmm0958":
                            map_location = torch.device('cpu')
                        print(exp_name)
                        decomposed_name = parse("{}/{}/{}/fold{:d}",exp_name)
                        exp_name_fold0 = os.path.join(decomposed_name[0],decomposed_name[1],"all_dms_datasets","fold0")
                        latest_checkpoint = os.path.join(log_dir, base_model, *(exp_name_fold0.split(os.path.sep)[1:]),"version_0","checkpoints","last.ckpt")
                        #num_training_steps += torch.load(latest_checkpoint,map_location="cpu",weights_only=False)["global_step"]
                        assert os.path.exists(latest_checkpoint), "Trying to fine tune a check point that does not exist: {}".format(latest_checkpoint)
                
                if latest_checkpoint is not None:
                    ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)

                    model = dmsEVE(ckpt["hyper_parameters"]["model_parameters"])

                    lEVE = lightningdmsEVE(model=model, hparams=model_params)

                    print("Restoring states from the checkpoint path at", latest_checkpoint)
                    #lEVE = lightningdmsEVE.load_from_checkpoint(latest_checkpoint, model=model, map_location="cpu")
                    lEVE.load_state_dict(ckpt["state_dict"],strict=True)
                    lEVE.epoch = ckpt["epoch"]

                    if (model_params["model_parameters"]["latent_to_dms_layer_sizes"] != lEVE.model.latent_to_dms.layers_sizes):
                        print("Initialize latent to DMS with new layers_sizes")
                        lEVE.model.init_latent_to_dms(model_params["model_parameters"])# = MLP(lEVE.model.latent_to_dms.input_size, model_params["model_parameters"]["latent_to_dms_layer_sizes"])
                    if "entropy" in model_params["model_parameters"]["vae_prior_type"]:
                        if model_params["model_parameters"]["latent_train_tau"] != lEVE.model.matVAE.tau.requires_grad:
                            lEVE.model.matVAE.tau.requires_grad_(model_params["model_parameters"]["latent_train_tau"])
                    
                else:
                    model = dmsEVE(model_params["model_parameters"])
                    lEVE = lightningdmsEVE(model=model, hparams=model_params)

                if verbose:
                    print(model)
                
                ##################################################################
                ####             LIGHTNING
                ##################################################################
                logger = TensorBoardLogger(log_dir, name=exp_name, default_hp_metric=False)
                os.makedirs(os.path.dirname(logger.log_dir), exist_ok=True)

                profiler = get_profilter(args.profiler)

                min_epochs = 1
                
                min_steps = training_parameters["min_steps"]#300#num_training_steps // 15
                enable_checkpointing = True
                if profiler:
                    num_training_steps = 3 * training_parameters["check_val_every_n_steps"] + 1
                    min_steps = num_training_steps
                    enable_checkpointing = False
                
                ### Checkpoints
                callbacks = []

                # Several checkpoints during training
                if enable_checkpointing:
                    chkpt_every_n_epochs = max([1,(num_training_steps // n_steps_per_epochs)//50])
                    checkpoint_callback = ModelCheckpoint(every_n_epochs=chkpt_every_n_epochs, verbose=verbose>0)
                    callbacks.append(checkpoint_callback)

                callbacks.append(ModelSummary(max_depth=2, verbose=verbose>0))

                limits = dict(limit_train_batches=None, limit_val_batches=None)
                
                # Check validation at least every epoch
                # Log validation metrics at most every epoch, at least every   check_val_every_n_steps/ n_steps_per_epochs    epochs
                check_val_every_n_epoch = max([1, training_parameters["check_val_every_n_steps"] // n_steps_per_epochs])

                if (socket.gethostname() == "cmm0958"):
                    enable_progress_bar = True
                    limits = dict(limit_train_batches=2, limit_val_batches=1)
                    num_training_steps = num_training_steps
                check_val_every_n_epoch = 2
                
                # Log training metric at most 10 times per epoch, and at least once
                log_every_n_steps = max([1, n_steps_per_epochs//10])
                if not( "dms" in data_parameters["training_data"]):
                    early_stop_callback = EarlyStopping(monitor="val_neg_ELBO", min_delta=training_parameters["min_delta"], patience=training_parameters["patience_n_steps"], mode="min",verbose=verbose>0)
                    callbacks.append(early_stop_callback)
                
                if verbose:
                    print("Training starting ...", "n_steps_per_epochs=", n_steps_per_epochs, "log_every_n_steps:", log_every_n_steps, "check_val_every_n_epoch",check_val_every_n_epoch)
                
                ##################################################################
                ####             FIT
                ##################################################################
                trainer = L.Trainer(deterministic=True, max_steps=num_training_steps, log_every_n_steps=log_every_n_steps,  devices=AVAIL_devices,
                                    check_val_every_n_epoch=check_val_every_n_epoch, enable_checkpointing=enable_checkpointing,
                                    enable_progress_bar=verbose > 1, **limits, callbacks=callbacks, logger=logger, profiler=profiler, 
                                    min_epochs=min_epochs, min_steps=min_steps)

                trainer.fit(lEVE,   train_dataloaders=train_dataloaders,
                                    val_dataloaders=val_loaders, 
                                   # ckpt_path=latest_checkpoint
                            )

                model_ckpt = checkpoint_callback.best_model_path

                last_checkpoint = os.path.join(logger.log_dir, "checkpoints", "last.ckpt")
                trainer.save_checkpoint(last_checkpoint)
                
                # Validate again on dms data for recording in separate files
                dmsloaders_validation = {k: v for i,(k, v) in  enumerate(val_loaders.items()) if k.startswith("dms_")}
                
                results = trainer.validate(lEVE, dataloaders=dmsloaders_validation)
                if len(results) > 1:
                    assert pd.DataFrame(results).drop_duplicates().shape[0] == 1, "Different values are reported for the same validation dataset. (If all the same, keep one copy.)"
                    results = [results[0]]
                results[0]["model/nb_trainable_params"] = sum(p.numel() for p in lEVE.parameters() if p.requires_grad)
                results[0]["model/nb_training_steps"] = lEVE.the_training_step
                # A list of validation dataloaders was passed probably. The fold averaging will only work if one dictionary of validation loaders is passed. "
                all_folds_results.append(results)

                plt.close("all")

                # Write the individual fold results
                write_pklz(
                    outputfname, 
                    results
                )
        
        # End iteration over folds
        ## Average and std over folds
        mean_dict = pd.DataFrame([ll[0] for ll in all_folds_results]).mean(0).to_dict()
        std_dict = pd.DataFrame([ll[0] for ll in all_folds_results]).std(0).to_dict()
        
        ## Write the result over all the folds in the common parent directory of each folds
        write_pklz(
            os.path.join(log_dir, os.path.dirname(exp_name), "results.pklz"), 
            [mean_dict,std_dict]
        )
        
        if not one_training_per_dms_dataset:#("msa" in data_parameters["training_data"]) and (not("dms" in data_parameters["training_data"])):
            break
    
    # End iteration over DMS datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored', default="./data/DMS_msa_files")
    parser.add_argument('--DMS_data_folder', type=str, help='Folder where MSAs are stored', default="./data/proteingym/ProteinGym_substitutions_DMS")

    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name', default="./data/mappings/example_mapping.csv")
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file', default=0)
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored', default="./data/DMS_msa_weights")
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored', default="./results/VAE_parameters")
    parser.add_argument('--model_name_suffix', default='Jan1_PTEN_example_lightning', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters', default="./logs/")
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--profiler', type=str, default=None)
    parser.add_argument("-i", '--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument("-j", type=int, default=1)
    parser.add_argument("-v", type=int, default=0)
    parser.add_argument("--retrain", action='store_true', help="Force retraining (default: False). By default lightning loads the latest ckpt file for the given configuration.", default=False)
    parser.add_argument("--nopin", action='store_true', help="pin_memory (default: True).", default=False)
    parser.add_argument("--persist", action='store_false', help="persist (default: True).", default=True)

    args = parser.parse_args()
    main(args)
    
