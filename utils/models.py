import torch
import sys, os
import math
import torch.nn.functional as F

from utils.mlp import MLP, init_all
from utils.activations import get_activation
from utils.matVAE import matVAE
from utils.data import get_aa_dict, read_pdb

from torch_geometric.utils import scatter, to_edge_index, dense_to_sparse
import json
import pandas as pd
import numpy as np


def get_grantham_map(fname):
    """ Transforms and stores a python dictionary equivalent to the tsv file used for Grantham Score computation.
    The original file is associated with this paper: 
        https://doi.org/10.1158/1055-9965.EPI-05-0469
    and the tsv file is stored on github:
        https://gist.github.com/danielecook/501f03650bca6a3db31ff3af2d413d2a
    """

    map_fname = os.path.join(os.path.dirname(fname), os.path.basename(fname).replace(".tsv",".json"))
    if not os.path.exists(map_fname):
        grantham_matrix = pd.read_csv(fname,sep="\t").set_index("FIRST")
        grantham_matrix.insert(0,"S",0)
        df_new_row = pd.DataFrame([pd.Series({aa:0 for aa in grantham_matrix.columns},name="W")])
        grantham_matrix = pd.concat([grantham_matrix,df_new_row],axis=0)
        assert(grantham_matrix.columns.tolist()==grantham_matrix.index.tolist())

        grantham_array = grantham_matrix.values
        grantham_array = np.tril(grantham_array.T,k=-1)+grantham_array
        grantham_matrix = pd.DataFrame(data= grantham_array,columns=grantham_matrix.columns,index=grantham_matrix.index)
        grantham_matrix_dict = {theindex: {thecol:str(grantham_matrix.values[i,j]) for j,thecol in enumerate(grantham_matrix.columns)} for i, theindex in enumerate(grantham_matrix.index)}
        
        # Add a missing residue
        grantham_matrix_dict["X"] = {**{k:np.nan for k in grantham_matrix_dict.keys()},**{"X":np.nan}}

        with open(map_fname,"w",encoding="utf8") as fp:
            json.dump(grantham_matrix_dict, fp)
        grantham_matrix.to_csv(map_fname.replace(".json",".csv"))
    else:
        with open(map_fname,"r",encoding="utf8") as fp:
            grantham_matrix_dict = json.load(fp)

        grantham_matrix = pd.read_csv(map_fname.replace(".json",".csv"))
        grantham_matrix.set_index("Unnamed: 0",inplace=True)

    grantham_matrix_dict = {aa1: {aa2:float(v) for aa2,v in grantham_matrix_dict[aa1].items()} for aa1 in grantham_matrix_dict.keys()}
    return grantham_matrix_dict,grantham_matrix


class dmsEVE(torch.nn.Module):
    def __init__(self, hparams):
        super(dmsEVE, self).__init__()
        self.hparams = hparams
        self.model_type = hparams["model_type"]
        self.reconstruction_error = hparams["reconstruction_error"]
        
        # Read & Preprocess grantham matrix
        self.init_grantham_matrix()

        # VAE
        self.matVAE = matVAE(hparams)
        
        # DMS
        self.init_latent_to_dms(hparams)
        
        self.internal_metrics = {}

    def init_latent_to_dms(self, hparams):
        self.latent_to_dms_detach = hparams["latent_to_dms_detach"]
        self.latent_to_dms_sigmoid = hparams["latent_to_dms_sigmoid"]
        self.latent_to_dms = MLP(self.matVAE.z_dim, hparams["latent_to_dms_layer_sizes"], hparams["latent_to_dms_output_size"], 
                                    get_activation(hparams["latent_to_dms_activation"]), dropout_p=hparams["latent_to_dms_dropout_p"],
                                    layernorm=hparams["layernorm"], skipconnections=hparams["skipconnections"], skiptemperature=hparams["skiptemperature"])
        
        init_all(self.latent_to_dms, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["latent_to_dms_activation"]))

    def __str__(self):
        model_string = "THE MODEL:\nData ({} x {})\n".format(self.hparams["L"], self.hparams["d"])

        if "transformers" in self.model_type:
            model_string += " -> Transformer ({} x {}) x{}\n".format(self.hparams["L"], self.hparams["d"], self.hparams["transformer_num_layers"])
        
        if "Lmax" in self.model_type:
            model_string += " -> Lmax ({}[={}])\n".format(", ".join(map(lambda i:str(i)+" x {}".format(self.hparams["d"]),
                                                     self.matVAE.mat_encoder_decoder_mu.MLP_down.layers_n_nodes)),
                                                     self.matVAE.mat_encoder_decoder_mu.MLP_down.layers_n_nodes[-1] * self.hparams["d"])
        model_string += " -> VAE ({}, {}[=z_dim])\n".format(", ".join(map(lambda l:str(l.in_features), self.matVAE.vae_encoder_mu.model.linear_layers)),self.hparams["vae_z_dim"])
        model_string += " -> DMS ({}, 1)\n".format(", ".join(map(str, self.latent_to_dms.layers_sizes)))
        return model_string

    def init_grantham_matrix(self):
        _, grantham_matrix = get_grantham_map("data/grantham.tsv")
        alphabet = get_aa_dict()

        grantham_matrix = grantham_matrix.rename(columns=alphabet).rename(alphabet).sort_index()
        grantham_matrix = grantham_matrix.reindex(sorted(grantham_matrix.columns), axis=1).replace({0:np.nan}).values
        
        min_grantham = np.nanmin(grantham_matrix)
        max_grantham = np.nanmax(grantham_matrix)

        grantham_matrix[np.isnan(grantham_matrix)]
        grantham_matrix = torch.from_numpy(grantham_matrix).to(torch.float)
        grantham_matrix[grantham_matrix==0] = torch.nan
        grantham_matrix = grantham_matrix/max_grantham
        grantham_matrix[grantham_matrix.isnan()] = 0
        self.register_buffer("grantham_matrix", grantham_matrix)

    def forward(self, x, stochastic_latent=True):
        N, L, d = x.shape

        z, latent_output = self.matVAE.encode(x, stochastic_latent=stochastic_latent)
        
        ## Predicting DMS
        #if False:
        if self.latent_to_dms_detach:
            y_hat_dms = self.latent_to_dms(latent_output["latent_to_dms_input"].detach())
        else:
            y_hat_dms = self.latent_to_dms(latent_output["latent_to_dms_input"])
        
        if self.latent_to_dms_sigmoid:
            y_hat_dms = F.sigmoid(y_hat_dms)

        latent_output["y_hat_dms"] = y_hat_dms

        logits = self.matVAE.decode(z)
        return logits, latent_output
    
    def compute_loss(self, x, y=None, beta=1, stochastic_latent=True, kl_global_params_scale=0):
        N = x.shape[0]
        logits, latent_output = self.forward(x, stochastic_latent=stochastic_latent)
        x_CE = 0.
        if y is None:

            x_CE = F.cross_entropy(logits.transpose(1,2), x.transpose(1,2), reduction="none").mean(1)#nll_loss(log_xhat.transpose(1,2), torch.argmax(x, dim=-1), reduction='none')

            # neg_ELBO
            neg_ELBO = latent_output["KLD"]*beta + x_CE
            if ("KLD_params" in latent_output.keys()) and (kl_global_params_scale>0):
                neg_ELBO += latent_output["KLD_params"]*kl_global_params_scale#self.training_parameters["kl_global_params_scale"]
            return neg_ELBO, latent_output, x_CE

        else:
            y_pred = latent_output["y_hat_dms"]#.flatten()
            loss =  F.mse_loss(y_pred[:,0], y[:,0], reduction="none").mean()

            return loss, latent_output
