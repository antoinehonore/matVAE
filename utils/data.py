

import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
import tqdm
from torch.utils.data import Dataset, Sampler, DistributedSampler, BatchSampler,DataLoader, Subset,ConcatDataset
from parse import parse 
import math
import random
from functools import partial
import biographs as bg
import networkx as nx
from torch_geometric.utils import scatter, to_edge_index, dense_to_sparse
import json
from utils_tbox.utils_tbox import write_pklz
from glob import glob

from sklearn.model_selection import train_test_split
DEFAULT_TYPE = torch.bool

def get_data_subset_indexes(thedataset, train_val_idxes, protein_name, dataset_name, subset=1, fold_idx=0, verbose=0):

    the_indexes = train_val_idxes[protein_name][dataset_name][fold_idx][subset]
    if verbose:
        print(protein_name, dataset_name, "subset={}, fold={}, #={}".format(subset, fold_idx, the_indexes.shape[0] if not (the_indexes is None) else None))
    
    if not (the_indexes is None):
        return Subset(thedataset, the_indexes) 
    else:
        return None

def get_data_subset(train_DMS_prot_datasets,train_val_idxes,subset=1,fold_idx=0,verbose=0):
    return {protein_name: 
                                {dataset_name: 
                                    get_data_subset_indexes(thedataset,train_val_idxes,protein_name,dataset_name, subset=subset, fold_idx=fold_idx,verbose=verbose)
                                    for dataset_name, thedataset in protein_datasets.items()
                                } for protein_name, protein_datasets in train_DMS_prot_datasets.items()
                        }

def get_train_val_idxes(n, validation_set_pct,random_state=None):
    all_indexes = torch.arange(n)
    if validation_set_pct == 0:
        out = [all_indexes, None]
    elif validation_set_pct == 1:
        out = [None, all_indexes]
    else:
        out = train_test_split(all_indexes,
                        test_size = validation_set_pct,
                        random_state=random_state)
    return out

def prepare_msa_dataloaders(all_MSA_datasets, training_proteins,test_proteins,num_workers,pin_memory, model_params, validation_set_pct=0.2,persistent_workers=False,random_state=None):
    training_parameters = model_params["training_parameters"]
    if training_proteins.to_frame()["protein_name"].shape[0]==1:
        model_params["model_parameters"]["protein_name"] = training_proteins.to_frame()["protein_name"][0]

    for i,k in enumerate(all_MSA_datasets.keys()):
        all_MSA_datasets[k].protein_name = k

    # Training/validation indexes
    train_val_idxes = {k: get_train_val_idxes(len(v),validation_set_pct,random_state=random_state) for k,v in all_MSA_datasets.items()}

    ## Get the training and validation subsets of the training proteins
    train_MSA_prot_dataset = {k: Subset(all_MSA_datasets[k],train_val_idxes[k][0]) for k in training_proteins}

    ## Create batch sampler taking into account the weights of the samples for the training loader
    batch_sampler = CustomBatchSampler(
        [len(d.indices) for d in train_MSA_prot_dataset.values()], 
        batch_size=training_parameters["batch_size"],
        weights=[d.dataset.weights[d.indices] for d in train_MSA_prot_dataset.values()]
        )

    ## Get the validation subsets of the training proteins
    val_MSA_prot_dataset = {k: Subset(all_MSA_datasets[k],train_val_idxes[k][1]) for k in training_proteins}
    
    ## Create protein specific validation loaders
    val_MSA_prot_loaders = {k:DataLoader(v,
            batch_size=training_parameters['batch_size'],
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
            shuffle=False) for k,v in val_MSA_prot_dataset.items()
    }

    # loaders for test proteins
    test_MSA_prot_dataset =  {k: all_MSA_datasets[k] for k in test_proteins}
    test_MSA_prot_loaders = {k:DataLoader(v,
            batch_size=training_parameters['batch_size'],
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,
            shuffle=False) for k,v in test_MSA_prot_dataset.items()
    }
    
    ## Create a sequential loader
    thedataset = ConcatDataset(train_MSA_prot_dataset.values())
    train_sequential_loader = DataLoader(thedataset,
                                            num_workers=get_num_workers(len(thedataset), training_parameters['batch_size'], num_workers), pin_memory=pin_memory,persistent_workers=persistent_workers,
                                            batch_sampler=batch_sampler)

    return train_MSA_prot_dataset, train_sequential_loader, val_MSA_prot_loaders, test_MSA_prot_loaders 

def get_num_workers(n, batch_size, max_workers=30):
    #n_batches = n // batch_size
    #return min([max_workers, max([0, n_batches])])
    return max_workers
    
def prepare_dms_dataloaders(train_DMS_prot_datasets, test_DMS_prot_datasets, num_workers, pin_memory, batch_size, fold_idx=0, train_val_idxes=None,verbose=0,persistent_workers=False):

    # Training/validation indexes
    # get_train_val_idxes(len(v[dataset_name]), validation_set_pct)
    #train_val_idxes = {protein_name: {dataset_name: list(idx_generator.split(np.arange(len(v[dataset_name]))))[validation_set_idx]
    #                                    for dataset_name in v.keys()
    #                                    }
    #                    for protein_name, v in train_DMS_prot_datasets.items()
    #                    }

    val_DMS_prot_datasets = get_data_subset(train_DMS_prot_datasets, train_val_idxes, subset=1,fold_idx=fold_idx)

    test_DMS_prot_datasets = {**val_DMS_prot_datasets, **test_DMS_prot_datasets}
    
    train_DMS_prot_datasets = get_data_subset(train_DMS_prot_datasets, train_val_idxes, subset=0,fold_idx=fold_idx,verbose=verbose)

    # Get dataloaders
    train_DMS_prot_dataloaders =  {"dms_" + os.path.basename(k): 
                                        DataLoader(v , batch_size=batch_size, 
                                        num_workers=get_num_workers(len(v), batch_size, max_workers=num_workers), shuffle=True, pin_memory=pin_memory,persistent_workers=persistent_workers)
                                        if not (v is None) else None
                                    for protein_name in train_DMS_prot_datasets.keys() 
                                    for k, v in train_DMS_prot_datasets[protein_name].items()
                                    }

    test_DMS_prot_dataloaders =  {"dms_" + os.path.basename(k): 
                    DataLoader(v , batch_size=batch_size, 
                    num_workers=get_num_workers(len(v),batch_size , max_workers=num_workers), shuffle=False, pin_memory=pin_memory,persistent_workers=persistent_workers)
                for protein_name in test_DMS_prot_datasets.keys() 
                for k, v in test_DMS_prot_datasets[protein_name].items()
            }

    return  train_DMS_prot_dataloaders, test_DMS_prot_dataloaders

class CustomBatchSampler(BatchSampler):
    r"""Yield a mini-batch of indices. 

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, sizes, batch_size, weights=None):
        super(CustomBatchSampler).__init__()
        # build data for sampling here
        self.batch_size = batch_size
        self.weights = weights
        if self.weights is None:
            self.indices = [torch.randperm(thesize).numpy().tolist() for thesize in sizes]
        else:
            self.indices = [torch.multinomial(theweights,thesize,replacement=True).numpy().tolist() for theweights,thesize in zip(self.weights,sizes)]
        self.list_len = [0] + np.cumsum(sizes).tolist()
        self.thelength = sum([math.ceil(thesize/self.batch_size) for thesize in sizes])
        #self.weights = []
        
    def __iter__(self):
        # implement logic of sampling here
        all_batches = []
        batch = []
        for ilist, indices in enumerate(self.indices):
        # torch.random.permutation(thesize)
            for i, item in enumerate(indices):
                batch.append(self.list_len[ilist]+item)
                
                if (len(batch) == self.batch_size) or (i == len(indices)-1):                    
                    all_batches.append(batch)
                    batch = []
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
        #print("")
    def __len__(self):
        return self.thelength

class TheDatasetDMS(Dataset):
    def __init__(self, x, y, x_wt=None, ybin=None, factorisation="none", 
        positional_encoding_type=None, positional_encoding_opts="",missing_target_distribution="default", adjacency_matrix=None):
        self.x = adjust_target_distribution(x,missing_target_distribution=missing_target_distribution)
        self.x_wt = adjust_target_distribution(x_wt,missing_target_distribution=missing_target_distribution)
        
        self.y = y
        self.y_rank = y.sort(descending=False).indices
        self.y_scaled = (y-y.min())/(y.max()-y.min())  ###.sort(descending=False).indices
        self.adjacency_matrix = adjacency_matrix
        self.ybin = ybin

        self.dummy = torch.empty(0)
        self.factorisation = factorisation

        self.x_wt_encoded = None
        self.x_wt_mat_reconstruction = None
        self.x_wt_mat_encoding = None
        self.x_wt_pos_encoding = None
        self.pos_encoding = None
        
        if self.factorisation != "none":
            x_encoded, mat_reconstruction, mat_encoding, pos_encoding, b = get_encoding_reconstruction(self.x,
                            factorisation=factorisation, positional_encoding_type=positional_encoding_type,positional_encoding_opts=positional_encoding_opts)

            self.x_mat_reconstruction = mat_reconstruction
            self.x_encoded = x_encoded
            self.pos_encoding = pos_encoding

            if not (x_wt is None):
                self.x_wt_encoded, self.x_wt_mat_reconstruction, x_wt_mat_encoding, self.x_wt_pos_encoding, b  = get_encoding_reconstruction(self.x_wt[None,...], 
                            factorisation=factorisation, positional_encoding_type=positional_encoding_type,positional_encoding_opts=positional_encoding_opts)
                
                self.wt_encoding_func_params = dict(mat_encoding=x_wt_mat_encoding,pos_encoding=pos_encoding, b=b)
                self.wt_decoding_func_params = dict(mat_reconstruction=self.x_wt_mat_reconstruction, pos_encoding=pos_encoding, b=b)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # Output dictionary keys
        # default:          x, DMS target
        # if x_wt:          x, DMS target, x_wt
        # if factorisation: x, u, y, target
        #   if x_wt:        x, u, y, target,   x_wt, u_wt, y_wt
        #   if factorisation and positional encoding was used, add pos_encoding
        
        out = {"x": self.x[idx]}
        #out["protein_name"] = self.protein_name
        #out["focus_cols"] = self.a_focus_cols

        out = {**out, "target": self.y[idx], "target_bin": self.ybin[idx], "target_rank": self.y_rank[idx], "target_rel_rank": self.y_rank[idx]/self.y_rank.shape[0],"target_scaled": self.y_scaled[idx]}

        if not (self.x_wt is None):
            out = {**out, "x_wt": self.x_wt}

        return out

class TheDataset(Dataset):
    def __init__(self, x, weights, x_wt=None,
                factorisation="none", flip_p=0,
                positional_encoding_type=None, positional_encoding_opts="", missing_target_distribution="default",adjacency_matrix=None):
        self.x = adjust_target_distribution(x, missing_target_distribution=missing_target_distribution)
        self.weights = weights
        self.flip_p = flip_p
        self.adjacency_matrix = adjacency_matrix

        N = self.x.shape[0]
        # Counts with Laplace smoothing
        # alpha=1e-5 (See Tranception: Protein Fitness Prediction with Autoregressive Transformers and Inference-time Retrieval)
        alpha = 1e-5
        d = self.x.shape[-1]

        N_i = torch.nansum(self.x, dim=0)
        N = self.x.shape[0]
        f_ca = (N_i + alpha)/(N + alpha*d)
        ### self.f_i = N_i / N

        num_different_residue_per_pos = (self.x.sum(0) > 0).sum(1).unsqueeze(-1)
        num_representation_per_residue_per_pos = self.x.sum(0)
        num_true_residue_per_pos = (self.x.sum(-1) == 1).sum(0)
        g_ca = 1 / (num_representation_per_residue_per_pos*num_different_residue_per_pos)
        g_ca[g_ca.isinf()] = 0

        gap_per_sequence_per_pos = self.x.sum(-1) == 0
        g_ca = g_ca + (gap_per_sequence_per_pos.sum(0) /N /20).unsqueeze(-1)
    
        self.x_wt = None

        if not (x_wt is None):
            self.x_wt = adjust_target_distribution(x_wt, missing_target_distribution=missing_target_distribution)

        self.factorisation = factorisation
        pos_encoding = None

        if self.factorisation != "none":
            x_encoded, mat_reconstruction,mat_encoding, pos_encoding, b = get_encoding_reconstruction(self.x, 
                            factorisation=factorisation,positional_encoding_type=positional_encoding_type,positional_encoding_opts=positional_encoding_opts)
            
            self.x_mat_reconstruction = mat_reconstruction
            self.x_encoded = x_encoded
            self.pos_encoding = pos_encoding

            if not (self.x_wt is None):
                x_wt_encoded, x_wt_mat_reconstruction, x_wt_mat_encoding, wt_pos_encoding, b = get_encoding_reconstruction(self.x_wt[None,...],
                                    factorisation=factorisation, positional_encoding_type=positional_encoding_type, positional_encoding_opts=positional_encoding_opts)
                assert((wt_pos_encoding == pos_encoding).all())
            

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.factorisation != "none":
            out =  {"x": self.x[idx], 
                    "mat_reconstruction":self.x_mat_reconstruction[idx],
                    "y": self.x_encoded[idx]} 
            
            if not (self.pos_encoding is None):
                out["pos_encoding"] = self.pos_encoding[idx]
        else:
            out = {"x": self.x[idx]} ###, self.u, self.svd_proj_matrix
            if np.random.uniform(0, 1) < self.flip_p:
                out["x"] = torch.flip(out["x"], dims=(0,))
        out["weight"] = self.weights[idx]
        out["protein_name"] = self.protein_name
        out["Neff"] = self.Neff
        out["focus_cols"] = self.a_focus_cols
        return out


def read_pdb(protein_name, root_dir="data/ProteinGym_AF2_structures",cutoff=10):
    pdb_fname = os.path.join(root_dir, "{}.pdb".format(protein_name))
    
    try:
        molecule = bg.Pmolecule(pdb_fname)
        pdb_network = molecule.network(cutoff=cutoff)
        adjacency_matrix = torch.from_numpy(nx.adjacency_matrix(pdb_network).todense()).to(torch.long)

    except:
        pdb_fname_wildcard = os.path.join(root_dir, "{}_*-*.pdb".format(protein_name))
        all_pdb_prot_files = sorted(glob(pdb_fname_wildcard),key=lambda s: parse("{}_{:d}-{:d}.pdb", os.path.basename(s))[1] )
        Adjacency = []

        for pdb_fname in  all_pdb_prot_files:
            molecule = bg.Pmolecule(pdb_fname)
            pdb_network = molecule.network(cutoff=10)
            adjacency_matrix = torch.from_numpy(nx.adjacency_matrix(pdb_network).todense()).to(torch.long)

            Adjacency.append(adjacency_matrix)
        adjacency_matrix = torch.block_diag(*Adjacency)
    return adjacency_matrix


def get_protein_data(protein_name, msa_location, weight_location, theta, data_parameters, training_parameters, all_dms_files, device="cpu", verbose=0,seed=None,num_workers=1,pin_memory=False,target_seq=None,mse_nlim=1e9):
    """ """
    data = MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            threshold_focus_cols_frac_gaps=1,
            weights_location=weight_location,
            nlim=mse_nlim
    )
    # Keep only focus columns
    if data.focus_start_loc>1:
        print("Start LOC", data.focus_start_loc)
        a_focus_cols = data.focus_start_loc + np.array(data.focus_cols) - 1
    else:
        a_focus_cols = np.array(data.focus_cols)
    
    focus_cols_outname = "data/ProteinGym_focuscols/{}.pklz".format(protein_name)
    write_pklz(focus_cols_outname,a_focus_cols)

    aa_dict = get_aa_dict()

    # Encoding function (onehot)
    fencode = partial(encode_sequence, aa_dict=aa_dict)
    
    MSA_data_tensor = torch.from_numpy(data.one_hot_encoding).to(dtype=DEFAULT_TYPE)

    if data_parameters["features"] == "noise":
        noise = torch.rand(thedata.shape[-1], thedata.shape[-1])
        # Make the codes random orthogonalvectors
        noise, _ = torch.linalg.qr(noise,mode="complete")
        MSA_data_tensor = thedata @ noise
    Neff = np.sum(data.weights)
    sampling_weights = torch.from_numpy(data.weights/Neff)

    if verbose:
        print(protein_name, "dataset sizes:")
        print("MSA data: ({},{},{})".format(*MSA_data_tensor.shape))
    
    adjacency_matrix = read_pdb(protein_name=protein_name,root_dir="data/ProteinGym_AF2_structures",cutoff=10)
    
    
    ###MSA_dataset.adjacency_matrix = adjacency_matrix
    
    MSA_dataset = TheDataset(MSA_data_tensor.to(device), sampling_weights.to(device),
                factorisation=data_parameters["factorisation_type"], #encoding_decoding_info=encoding_decoding_info,
                positional_encoding_type=data_parameters["positional_encoding_type"],
                positional_encoding_opts=data_parameters["positional_encoding_opts"],
                flip_p=training_parameters['data_flip_p'],
                missing_target_distribution=data_parameters["missing_target_distribution"],
                adjacency_matrix=adjacency_matrix)
    
    MSA_dataset.protein_name = protein_name
    MSA_dataset.Neff = Neff

    prot_dms_files = [fname for fname in all_dms_files if os.path.basename(fname).startswith(protein_name)]
    if verbose:
        print(protein_name, ": associated DMS", prot_dms_files)
    

    MSA_dataset.a_focus_cols = torch.from_numpy(a_focus_cols)

    DMS_datasets = {}

    for dms_fname in prot_dms_files:
        dms_df = pd.read_csv(os.path.join(".", dms_fname))

        dms_df["mutated_sequence"] = dms_df["mutated_sequence"].apply(lambda s: np.array(list(s))[a_focus_cols])
        if not dms_df.empty:

            DMS_ref_wt_seq = "".join(np.array(list(target_seq))[a_focus_cols].tolist())# DMS_ref_df.loc[DMS_ref_df["DMS_filename"] == os.path.basename(dms_fname),"target_seq"]
            MSA_ref_wt_seq = "".join(data.seq_name_to_sequence[data.focus_seq_name])

            assert(DMS_ref_wt_seq == MSA_ref_wt_seq)

            x_wt = torch.from_numpy(fencode(DMS_ref_wt_seq)).to(DEFAULT_TYPE)[0]
            
            dms_data_tensor = np.concatenate(dms_df["mutated_sequence"].apply(fencode).values.tolist())
            dms_target_tensor = dms_df["DMS_score"].values
            dms_bintarget_tensor = dms_df["DMS_score_bin"].values

            X = torch.from_numpy(dms_data_tensor).to(DEFAULT_TYPE)
            Y = torch.from_numpy(dms_target_tensor).to(torch.float)
            # Normalization used here> https://doi.org/10.1016/j.cels.2023.07.003
            # " assumption that mutations that result in a complete loss of function are comparable among different DMS datasets."
            Y = Y / torch.quantile(Y, 0.05).abs()

            Ybin = torch.from_numpy(dms_bintarget_tensor).to(DEFAULT_TYPE)

            if data_parameters["features"] == "noise":
                X = X @ noise
                x_wt = x_wt @ noise
            
            print("DMS: ({},{},{})".format(*X.shape))

            dataset_te_dms = TheDatasetDMS(X.to(device), y=Y.to(device), x_wt=x_wt.to(device), ybin=Ybin.to(device),
                            factorisation=data_parameters["factorisation_type"], #encoding_decoding_info=encoding_decoding_info,
                            positional_encoding_type=data_parameters["positional_encoding_type"],
                            positional_encoding_opts=data_parameters["positional_encoding_opts"],
                            missing_target_distribution=data_parameters["missing_target_distribution"],
                            adjacency_matrix=adjacency_matrix)
            dataset_te_dms.protein_name = protein_name
            dataset_te_dms.a_focus_cols = torch.from_numpy(a_focus_cols)

            DMS_datasets[dms_fname] = dataset_te_dms
    
    return MSA_dataset, DMS_datasets


# Replace with WT encoding/decoding
def replace_encoding_with_WT(thedataset,decoding_params,encoding_params):    
    thedataset.x_mat_reconstruction = decoding_params["mat_reconstruction"].repeat((thedataset.x_mat_reconstruction.shape[0],1,1))
    thedataset.x_encoded = encoding_function(thedataset.x, **encoding_params)
    return thedataset

def get_encoding_reconstruction(X, factorisation="SVD", positional_encoding_type=None, positional_encoding_opts=None):
    """ Input (N,L,d) tensor."""
    X_pos, pos_encoding, b = positional_encoding(X,
                positional_encoding_type=positional_encoding_type,positional_encoding_opts=positional_encoding_opts)
    
    if factorisation == "SVD":
        X_encoded, mat_reconstruction, mat_encoding = svd_proj(X_pos)
    
    elif factorisation == "QR":
        X_encoded, mat_reconstruction, mat_encoding = qr_proj(X_pos)
    elif factorisation == "DFT":
        X_encoded, mat_reconstruction, mat_encoding = dft_proj(X_pos)
    decoding_error = ((mat_reconstruction @ X_encoded) - positional_encoding(X,   
                                                                positional_encoding_type=positional_encoding_type,
                                                                positional_encoding_opts=positional_encoding_opts)[0]).abs().max()
    print("Decoding error: ", decoding_error)
    return X_encoded, mat_reconstruction, mat_encoding, pos_encoding, b

def encoding_function(X, mat_encoding=None, pos_encoding=None, b=None):
    Xout = X
    
    if not (b is None):
        Xout = torch.cat([b.repeat((Xout.shape[0], 1, 1)), Xout], dim=-1)
    if not (pos_encoding is None):
        Xout = Xout + pos_encoding
    
    if not (mat_encoding is None):
        Xout = mat_encoding @ Xout
    return Xout

def decoding_function(X, mat_reconstruction=None, pos_encoding=None, b=None):
    Xout = X
    
    if not (mat_reconstruction is None):
        Xout = mat_reconstruction @ Xout
    
    if not (b is None):
        Xout = Xout[:,:,b.shape[2]:]
    
    if not (pos_encoding is None):
        Xout = Xout - pos_encoding
    return Xout

def adjust_target_distribution(X, missing_target_distribution="default"):
    """Input tensor of shape (N,L,d). The discrete distribution is assumed over the last dimension.""" 
    if missing_target_distribution  == "uniform":
        X[(X.sum(-1) == 0)] = 1 / X.shape[-1]
    return X

def get_aa_dict():
    """Declare the alphabet"""
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {}
    for i, aa in enumerate(alphabet):
        aa_dict[aa] = i
    return aa_dict

def encode_sequence(s, aa_dict=None):
    out = np.zeros((len(s),len(aa_dict)))
    alphabet=aa_dict.keys()
    for i in range(len(s)):
        if s[i] in alphabet:
            out[i, aa_dict[s[i]]]=1
    return out[None,...]

def filter_from_msa(dms_df, focus_cols):
    #a_focus_cols = np.array(focus_cols)
    #row_idx = np.array([parse("{}{:d}{}",s)[1]-1 in focus_cols for s in dms_df["mutant"]])
    #out = dms_df[row_idx].copy()
    
    return out

def get_wt_seq(dms_df):
    first_row = dms_df.iloc[0]
    wt_seq = list(first_row.mutated_sequence)
    mutant = first_row.mutant
    
    assert(not (":" in mutant))
    m,i,v = parse("{}{:d}{}",mutant)

    assert(wt_seq[i-1]==v)
    wt_seq[i-1]=m
    
    return "".join(wt_seq)

def positional_encoding(Xall, alpha=1, positional_encoding_type=None, positional_encoding_opts=""):
    
    Xall_cat = Xall
    pos_encoding = None #### torch.zeros_like(Xall_cat)
    b = None
    Xall_pos = Xall
    if (positional_encoding_type=="cos/sin"):
        t = torch.arange(Xall.shape[1])[None,:,None]/Xall.shape[1]
        b = torch.cat([torch.cos(2*torch.pi*t)*alpha, torch.sin(2*torch.pi*t)*alpha],dim=-1).repeat((Xall.shape[0],1,1))

        if positional_encoding_opts == "norm_2nbits":
            Xall_cat = Xall_cat / 2

        Xall_pos = torch.cat([b.to(Xall.device), Xall_cat], dim=-1)

    elif (positional_encoding_type == "binary"):
        n_bits = 10  #int(np.ceil(np.log2(Xall.shape[1])))#10
        assert ( (2**n_bits) > Xall.shape[1])
        b = torch.tensor([list(map(int, list(format(tt, '#0{}b'.format(n_bits+2)))[2:])) for tt in range(1, Xall.shape[1]+1)]).unsqueeze(0)
        # b = b / b.sum(-1)[...,None]
        # b[b.isnan()] = 0
        b = b.repeat((Xall.shape[0], 1, 1))

        #if not (positional_encoding_opts == ""):
        if positional_encoding_opts == "norm_nbits":
            b = b / n_bits
        elif  positional_encoding_opts == "norm_2nbits":
            b = b / (2*n_bits)
        
        if positional_encoding_opts == "norm_2nbits":
            Xall_cat = Xall_cat/2

        Xall_pos = torch.cat([b.to(Xall.device), Xall_cat], dim=-1)

    elif positional_encoding_type == "attention":
        _, L, d = Xall_cat.shape
        pos = torch.arange(L).reshape(L, 1)
        all_ds = torch.arange(d).reshape(1,d)
        MAX_LENGTH = 10000
        
        sinus = torch.sin(pos*(MAX_LENGTH**(-all_ds/d))).unsqueeze(0)
        cosinus = torch.cos(pos*(MAX_LENGTH**(-(all_ds-1)/d))).unsqueeze(0)

        pos_encoding = (torch.where((all_ds.reshape(-1) % 2) == 0, sinus, cosinus)/((d/2)**0.5)).repeat((Xall_cat.shape[0], 1, 1))

        Xall_pos = Xall_cat + pos_encoding
    
    elif positional_encoding_type == "attention_cat":
        _, L, d = Xall_cat.shape
        d = 6 #
        pos = torch.arange(L).reshape(L, 1)
        all_ds = torch.arange(d).reshape(1,d)
        MAX_LENGTH = 10000
        
        sinus = torch.sin(pos*(MAX_LENGTH**(-all_ds/d))).unsqueeze(0)
        cosinus = torch.cos(pos*(MAX_LENGTH**(-(all_ds-1)/d))).unsqueeze(0)

        b = torch.where((all_ds.reshape(-1) % 2) == 0, sinus, cosinus)/((d/2)**0.5)

        Xall_pos = torch.cat([b.repeat((Xall_cat.shape[0], 1, 1)), Xall_cat],dim=-1)

    elif (positional_encoding_type == "simple"):
        #n_bits = 10  #int(np.ceil(np.log2(Xall.shape[1])))#10
        #assert ( (2**n_bits) > Xall.shape[1])
        b = (torch.arange(Xall.shape[1])+1)/Xall.shape[1] #tensor([list(map(int, list(format(tt, '#0{}b'.format(n_bits+2)))[2:])) for tt in range(1, Xall.shape[1]+1)]).unsqueeze(0)
        #b = b / b.sum(-1)#[...,None]
        # b[b.isnan()] = 0
        b = b[None,:,None].repeat((Xall.shape[0], 1, 1))
        Xall_pos = torch.cat([b.to(Xall.device), Xall_cat], dim=-1)

    elif (positional_encoding_type == "diff"):
        n = 3
        xdiff = Xall.diff(n,dim=1, prepend=Xall[:,-n:,:]).roll(-n+1,dims=1)
        pos_encoding = (xdiff - xdiff.min())/(xdiff.max()-xdiff.min())
        Xall_pos = Xall_cat + pos_encoding
        
        if False:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            im = axes[0].imshow(Xall_pos[0,:37], aspect="auto");plt.colorbar(im)
            im = axes[1].imshow(Xall_cat[0][:37], aspect="auto");plt.colorbar(im)
    
    if not (b is None):
        b = b[:1]
    
    return Xall_pos, pos_encoding, b

def svd_proj(X, k=None):
    """ X: (N,L,D);
        X_proj: (N,D,D);   
        mat_reconstruction @ X_proj == X;
        mat_encoding @ X == X_proj
    """
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    assert k is None, "Not yet implemented"
    
    ###### Proj = torch.diag_embed(S) @ Vh
    # if ((X>=0).all()) and (X.shape[-1]==20): 
    #     Vh = Vh.abs()
    #     U = U.abs()
    #######################################

    X_proj = Vh[:, :k, :]
    mat_reconstruction = U[:,:,:S.shape[1]] @ torch.diag_embed(S[:,:k])
    mat_encoding = torch.diag_embed(S[:,:k])**(-1) @ U[:, :, :k].transpose(1,2)
    return X_proj, mat_reconstruction, mat_encoding

def qr_proj(X):
    Q, R = torch.linalg.qr(X, mode="reduced")
    X_proj = R                                ####[:,:R.shape[-1],:]  ####R
    mat_reconstruction = Q
    mat_encoding = Q.transpose(1, 2)
    return X_proj, mat_reconstruction, mat_encoding

def inverse_dft_matrix(N):
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    F_inv = np.conj(F.T) 
    return F_inv

def dft_proj(X, N=100):
    """Abandon sur le tour de France"""
    L=X.shape[1]
    I = torch.eye(L)
    x = X[0,:,0]

    plt.close("all")

    n = 300
    a = torch.randn(L, n)
    xfft = torch.fft.fft(x, n=n)
    xfft = x[None, :] @ a
    
    b=torch.linalg.pinv(a)
    xhat = torch.fft.ifft(xfft, n=L)
    xhat = xfft @ b

    print(torch.linalg.norm(x - xhat)**2)

    fig, ax = plt.subplots()
    ax.plot(x,color="black",lw=3)
    ax.plot(xhat.reshape(-1))
    plt.show()

    iprojection = projection.conj().T 
    
    mat_reconstruction = iprojection

    X_proj = (X.transpose(1,2).cfloat() @ projection)


    X_proj_cat = torch.cat([X_proj.real, X_proj.imag], dim=2)
    
    import matplotlib; 
    matplotlib.use("tkagg")
    import matplotlib.pyplot as plt

    return X_proj, mat_reconstruction, mat_encoding

def utv_decomp(X,plot=False):
    U, R = torch.linalg.qr(X,           mode="reduced")
    R_t = R.transpose(1,2)
    V_prime, T_t = torch.linalg.qr(R_t, mode="reduced")
    Vh = V_prime.transpose(1,2)
    T = T_t.transpose(1,2)
    
    X_proj = Vh
    mat_reconstruction = U @ T
    mat_encoding = T.transpose(1,2) @ U.transpose(1, 2)
    
    if plot:
        print(((U @ T @ Vh - Xall_pos)**2).abs().max())
        isample=0
        fig,axes=plt.subplots(2,2,figsize=(10,10))
        axes[0,0].imshow(R[isample],aspect="auto")
        axes[0,0].set_title("R (A=UR)")
        axes[0,1].imshow(T[isample],aspect="auto")
        axes[0,1].set_title("T $(R^T = V T^T)$")
        axes[1,0].imshow(X_proj[isample],aspect="auto")
        axes[1,0].set_title("$A_{proj}=V^T$")
        axes[1,1].imshow(Vh[isample],aspect="auto")
        axes[1,1].set_title("$V^T$")
    return X_proj, mat_reconstruction, mat_encoding


class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True,
        nlim=None
        ):
        
        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols
        self.nlim=nlim
        self.gen_alignment()
        self.create_all_singles()

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    if len(self.seq_name_to_sequence)>self.nlim:
                        print("")
                        break
                    name = line
                    if i == 0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.iloc[seq_idx]["sequence"]#.sequence[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; keeps focus columns only
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        print ("Encoding sequences")
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                
                print("Loaded sequence weights from disk")
            except:
                print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.theta) 
                        return 1/denom
                    else:
                        return 0.0 #return 0 weight if sequence is fully empty
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
            if not( self.weights.shape[0] == self.one_hot_encoding.shape[0]):#, "Weights and MSA do not have the same number of sequences."
                self.weights = np.ones(self.one_hot_encoding.shape[0])

        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.one_hot_encoding.shape)
    
    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i,letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx+i
                    if mut != letter:
                        mutant = letter+str(pos)+mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1   
        self.all_single_mutations = list_valid_mutations

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)



class MSA_processing_vectorized:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=0.3,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):
        
        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols
        
        self.gen_alignment()
        self.create_all_singles()
        assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
        assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        
        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df["sequence"] = msa_df["sequence"].apply(lambda s: list(s.replace(".","-").upper())) ## msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))
            
            msa_array = np.array(msa_df['sequence'].apply(lambda a: a).values.tolist())  #np.array([list(seq) for seq in msa_df.sequence])
            msa_seq_name = msa_df.index.values
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = msa_array[0] !='-'#[aa!='-' for aa in msa_df.sequence_a[self.focus_seq_name]]
            msa_array = msa_array[:,non_gap_wt_cols]
            #msa_df['sequence_a'] = msa_df['sequence_a'].apply(lambda x: x[non_gap_wt_cols] )
            
            #msa_array = np.array(msa_df['sequence_a'].apply(lambda a:a.tolist()).values.tolist())  #np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = msa_array == "-"  #np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))
            
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            
            # Lower case non focus cols and filter fragment sequences
            msa_array = msa_array[seq_below_threshold]
            msa_seq_name = msa_seq_name[seq_below_threshold]


            #msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            #msa_df = msa_df[seq_below_threshold]
            
            # Overwrite seq_name_to_sequence with clean version

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        
        self.focus_cols = np.argwhere( 
            (index_cols_below_threshold) & (msa_array[list(msa_seq_name).index(self.focus_seq_name)]!="-")
            ).reshape(-1)  ##[ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 

        self.focus_seq_trimmed = "".join(np.array(list(self.focus_seq))[self.focus_cols])  ###self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start,stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 


        # Move all letters to CAPS; keeps focus columns only
        msa_array = msa_array[:,self.focus_cols]
        #self.seq_name_to_sequence = {seq_name:[sequence[ix].upper() for ix in self.focus_cols] for seq_name,sequence in self.seq_name_to_sequence.items()}
        
        #for seq_name,sequence in self.seq_name_to_sequence.items():
        #    sequence = sequence.replace(".","-")
        #    self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        #def dummy(s):
        #    print(s)
        #    return "a" in s
        
        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_w_indeterminate_AA = ((msa_array != "-") & ~np.isin(msa_array, list(self.alphabet))).sum(1)>0
            assert not seq_w_indeterminate_AA[0], "The WT seq has indeterminate AAs"
            msa_array = msa_array[~seq_w_indeterminate_AA]
            msa_seq_name = msa_seq_name[~seq_w_indeterminate_AA]
            #self.seq_name_to_sequence = {seq_name:sequence for seq_name,sequence in self.seq_name_to_sequence.items() 
            #                                if not any(
             #                                   letter not in alphabet_set and letter != "-" for letter in sequence
            #                                    )}
            #for seq_name,sequence in self.seq_name_to_sequence.items():
            #    for letter in sequence:
            #        if letter not in alphabet_set and letter != "-":
            #            seq_names_to_remove.append(seq_name)
            #            continue
            #seq_names_to_remove = list(set(seq_names_to_remove))
            #for seq_name in seq_names_to_remove:
            #    del self.seq_name_to_sequence[seq_name]

        self.seq_name_to_sequence = defaultdict(str)
        for seq_idx in range(msa_seq_name.shape[0]):
            self.seq_name_to_sequence[msa_seq_name[seq_idx]] = "".join(msa_array[seq_idx])
        
        self.aa_dict["-"] = len(self.aa_dict)
        array = msa_array.copy()
        # Replace characters using np.where and a loop through the dictionary
        
        for char, digit in self.aa_dict.items():
            msa_array = np.where(msa_array == char, digit, msa_array)
        msa_array = msa_array.astype(int)
        ##print(array)
        initial_size = msa_array.shape

        self.one_hot_encoding = np.zeros((msa_array.size, len(self.aa_dict)),bool)
        self.one_hot_encoding[np.arange(msa_array.size), msa_array.reshape(-1)]=True
        assert (self.one_hot_encoding.sum(1)==1).all()
        self.one_hot_encoding = self.one_hot_encoding.reshape(*initial_size,-1)[...,:-1]
        del self.aa_dict["-"]

        # Encode the sequences
        #print ("Encoding sequences")
        #self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)))
        #onehot_codes = {k: np.zeros(len(self.alphabet)) for k in self.aa_dict}
        #onehot_codes["-"] =  np.zeros(len(self.alphabet))
        #for k,i in self.aa_dict.items():
        #    onehot_codes[k][i] = 1.0
        
        #for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
        #    sequence = self.seq_name_to_sequence[seq_name]

        #    self.one_hot_encoding[i] = np.concatenate([onehot_codes[k][None,:] for k in sequence])
            #for j,letter in enumerate(sequence):
            #    if letter in self.aa_dict: 
            #        k = self.aa_dict[letter]
            #        self.one_hot_encoding[i,j,k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                print("Loaded sequence weights from disk")
            except:
                print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))

                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq, seq) / number_non_empty_positions#np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.theta) 
                        return 1/denom
                    else:
                        return 0.0 #return 0 weight if sequence is fully empty

                nlim = 1000
                K = list_seq[:nlim] @ list_seq[:nlim].T
                number_non_empty_positions = (list_seq[:nlim]**(2)).sum(1)[:,None]#.unsqueeze(1)
                
                self.weights = np.array(list(map(compute_weight, list_seq[:nlim])))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]

        print ("Neff =",str(self.Neff))
        print ("Data Shape =",self.one_hot_encoding.shape)

    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i,letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx+i
                    if mut != letter:
                        mutant = letter+str(pos)+mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1   
        self.all_single_mutations = list_valid_mutations
    
    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)