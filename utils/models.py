import torch
import sys, os
import torch.nn as nn
import math
import torch.nn.functional as F

from utils.decoders import VAE_Bayesian_MLP_decoder, simple_mlp_decoder
from utils.encoders import simple_mlp_encoder
from utils.mlp import create_nn_sequential_MLP,MLP
from utils.activations import get_activation
from utils.attention import MultiSelfAttention
from utils.priors import MogPrior, VampPrior, kl_divergence_two_gaussians,kl_divergence_gaussian_vs_mog,rho_to_logvar

from utils.data import get_aa_dict, read_pdb

from torch_geometric.utils import scatter, to_edge_index, dense_to_sparse
import biographs as bg
import networkx as nx
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



class VarLenMLPup(torch.nn.Module):
    def __init__(self, max_size, insize, layers_sizes, activation, dropout_p=0, layernorm=False,skipconnections=False):
        super(VarLenMLPup, self).__init__()
        self.layernorm = layernorm
        self.layers_sizes = layers_sizes + [1]
        self.insize = insize
        self.max_size = max_size
        self.skipconnections = skipconnections

        self.layers_n_nodes = [max([int(max_size*self.layers_sizes[i]), insize]) for i in range(len(self.layers_sizes))]
        self.dropout_p=dropout_p
        self.dropout=torch.nn.Dropout(dropout_p)
        self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(insize, self.layers_n_nodes[0])] + \
                [torch.nn.Linear(self.layers_n_nodes[i-1], self.layers_n_nodes[i]) for i in range(1, len(self.layers_sizes))]
            )
        self.activation = activation()
        self.current_L = None
        self.internal_metrics = {}

    @torch.compile(mode="default")
    def forward(self, x):
        assert(not self.current_L is None)
        L = self.current_L
        y = x
        
        for i in range(len(self.layers)):
            if i == 0:
                layer_in_size = None ###self.insize
            else:
                layer_in_size = max([self.insize, int(L * self.layers_sizes[i-1])])  ###### max([self.outsize, int(L*self.layers_sizes[i])])
            
            
            if i == len(self.layers)-1:
                layer_out_size = int(L * self.layers_sizes[i])
            else:
                layer_out_size = max([self.insize,  int(L * self.layers_sizes[i])])

            ###print(i, layer_in_size, "->", layer_out_size)
            A_l_i = self.layers[i].weight[:layer_out_size, :layer_in_size]
            bias_l_i = self.layers[i].bias[:layer_out_size]
            
            y = F.linear(   
                y.transpose(1, 2), weight=A_l_i, bias=bias_l_i
            ).transpose(1, 2)
            
            if i < len(self.layers) - 1:
                y = self.activation(y)
                y = self.dropout(y)
                if self.layernorm and (y.ndim > 1):
                    y = torch.nn.functional.layer_norm(y, normalized_shape=[y.shape[-1]])
        return y

class VarLenMLPdown(torch.nn.Module):
    def __init__(self, max_size, outsize, layers_sizes, activation, dropout_p=0, layernorm=False, skipconnections=False):
        super(VarLenMLPdown, self).__init__()
        self.layernorm=layernorm
        self.layers_sizes = [1] + layers_sizes
        self.outsize = outsize
        self.skipconnections = skipconnections
        self.max_size = max_size
        self.layers_n_nodes = [max([int(max_size*self.layers_sizes[i]), outsize]) for i in range(len(self.layers_sizes))]
        self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(self.layers_n_nodes[i], self.layers_n_nodes[i+1]) for i in range(len(self.layers_sizes)-1)]+\
                [torch.nn.Linear(self.layers_n_nodes[-1], outsize)]
            )
        self.dropout_p=dropout_p
        self.dropout=torch.nn.Dropout(dropout_p)
        ##self.linear_down = torch.nn.Linear(self.Lmax, self.H)
        ##self.linear_up = torch.nn.Linear(self.H, self.Lmax)
        self.activation = activation()
    
    @torch.compile(mode="default")
    def forward(self, x):
        L = x.shape[1]
        y = x
        if len(self.layers) > 0:
            for i in range(len(self.layers)):
                
                layer_in_size = int(L*self.layers_sizes[i])
                if i > 0:
                    layer_in_size = max([self.outsize,layer_in_size])
                
                if i < len(self.layers) - 1:
                    layer_out_size = max([self.outsize, int(L*self.layers_sizes[i+1])])
                else:
                    #Take all the weights in the output layer
                    layer_out_size = None
                
                ##print(layer_in_size, "->", layer_out_size)
                A_l_i = self.layers[i].weight[:layer_out_size, :layer_in_size]
                bias_l_i = self.layers[i].bias[:layer_out_size]
                
                y = F.linear(   
                    y.transpose(1, 2), weight=A_l_i, bias=bias_l_i
                ).transpose(1, 2)

                if i < len(self.layers) - 1:
                    y = self.activation(y)
                    y = self.dropout(y)
                    if self.layernorm and (y.ndim > 1):
                        y = torch.nn.functional.layer_norm(y, normalized_shape=[y.shape[-1]])
            
        return y


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        if p.requires_grad:
            #rand_int=torch.randint(-2,2,()).item()
            #rand_int=(torch.rand(1).item())*2-1
            #init_func(p, *params, **{k:v+rand_int for k,v in kwargs.items()})
            if p.ndim > 1:# Weight tensors
                init_func(p, *params, **kwargs)
            else: # Biases
                torch.nn.init.constant_(p,0.01)
            #p.data = torch.nn.functional.normalize(p,dim=-1).data
            #torch.nn.utils.weight_norm(p,name='weight',dim=1)



class dmsEVE(torch.nn.Module):
    def __init__(self, hparams):
        super(dmsEVE, self).__init__()
        self.hparams = hparams

        self.model_type = hparams["model_type"]
        self.layernorm = hparams["layernorm"]
        self.skipconnections = hparams["skipconnections"]
        if "skiptemperature" in hparams.keys():
            self.skiptemperature = hparams["skiptemperature"]
        else:
            self.skiptemperature=1

        self.latent_to_dms_detach = hparams["latent_to_dms_detach"]
        self.latent_to_dms_sigmoid = hparams["latent_to_dms_sigmoid"]

        self.reconstruction_error = hparams["reconstruction_error"]

        self.include_temperature_scaler = hparams["include_temperature_scaler"]
        
        # Read protein structure
        
        self.init_adjacency(hparams["protein_name"], hparams["transformer_use_structure"])
        

        # Read & Preprocess grantham matrix
        self.init_grantham_matrix()

        # TRANSFORMER 
        if "transformers" in self.model_type:
            self.init_transformers(hparams)

        # VAE
        self.init_vae(hparams)
        
        # DMS
        self.latent_to_dms_layer_sizes = hparams["latent_to_dms_layer_sizes"]
        self.latent_to_dms = MLP(self.z_dim, self.latent_to_dms_layer_sizes, 1, get_activation(hparams["latent_to_dms_activation"]), dropout_p=0,
             layernorm=self.layernorm,skipconnections=self.skipconnections, skiptemperature=self.skiptemperature)
        init_all(self.latent_to_dms, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["latent_to_dms_activation"]))
        
        # LMAX network
        if "Lmax" in self.model_type:
            self.init_variable_length(hparams)

        if self.include_temperature_scaler > 0:
            #self.temperature_scaler_mean = 
            #self.register_buffer("temperature_scaler_mean", self.include_temperature_scaler*torch.ones(1))
            self.temperature_scaler_mean = torch.nn.Parameter(self.include_temperature_scaler*torch.ones(1))

        self.internal_metrics = {}

    def __str__(self):
                
        model_string = "THE MODEL:\nData ({} x {})\n".format(self.hparams["L"], self.hparams["d"])

        if "transformers" in self.model_type:
            model_string += " -> Transformer ({} x {}) x{}\n".format(self.hparams["L"], self.hparams["d"], self.hparams["transformer_num_layers"])
        
        if "Lmax" in self.model_type:
            model_string += " -> Lmax ({}[={}])\n".format(", ".join(map(lambda i:str(i)+" x {}".format(self.hparams["d"]), self.MLP_down.layers_n_nodes)), self.MLP_down.layers_n_nodes[-1] * self.hparams["d"])
        model_string += " -> VAE ({}, {}[=z_dim])\n".format(", ".join(map(lambda l:str(l.in_features), self.vae_encoder.model.linear_layers)),self.hparams["vae_z_dim"])
        model_string += " -> DMS ({}, 1)\n".format(", ".join(map(str, self.latent_to_dms_layer_sizes)))
        return model_string

    def init_adjacency(self, protein_name, cutoff):
        if cutoff > 0:
            adjacency_matrix = read_pdb(protein_name=protein_name, root_dir="data/ProteinGym_AF2_structures",cutoff=cutoff)
            focus_cols_fname = os.path.join("data", "ProteinGym_focuscols", "{}.pklz".format(protein_name))
            from utils_tbox.utils_tbox import read_pklz
            focus_cols = torch.from_numpy(read_pklz(focus_cols_fname))
            if (adjacency_matrix==False).all():
                print(protein_name, "increase cutoff to",cutoff+1)
                pdb_network = molecule.network(cutoff=cutoff+1)
                adjacency_matrix = torch.from_numpy(nx.adjacency_matrix(pdb_network).todense()).to(torch.long)
            adjacency_matrix = adjacency_matrix[focus_cols][:, focus_cols]
            self.register_buffer("adjacency_matrix", (adjacency_matrix > 0))
        else:
            self.adjacency_matrix = None
        
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

    def init_variable_length(self,hparams):
        self.Lmax = hparams["variable_length_Lmax"]
        self.H = hparams["variable_length_H"]

        n_layers = hparams["variable_length_n_layers"]
        
        variable_length_layers = [i / (n_layers) for i in range(1, n_layers)][::-1]

        self.MLP_down = VarLenMLPdown(self.Lmax, self.H, variable_length_layers,   
                        get_activation(hparams["variable_length_activation"]), 
                        dropout_p=hparams["variable_length_dropout_p"],layernorm=self.layernorm,skipconnections=self.skipconnections
                        )
        init_all(self.MLP_down, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["variable_length_activation"]))

        self.MLP_up = VarLenMLPup(self.Lmax, self.H, variable_length_layers[::-1], 
                        get_activation(hparams["variable_length_activation"]), 
                        dropout_p=hparams["variable_length_dropout_p"],layernorm=self.layernorm,skipconnections=self.skipconnections
                        )
        
        init_all(self.MLP_up, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["variable_length_activation"]))

    def init_latent(self, hparams):
            
        self.prototype_shape = (hparams["L"],hparams["d"])

        self.prior_type = hparams["vae_prior_type"]
        self.num_prior_components = hparams["vae_num_prior_components"]

        if "mog" in self.prior_type:
            self.Prior = MogPrior(self.z_dim, self.num_prior_components, self.prior_type, init_scaler=self.prior_init_scaler)

        elif "vamp" in self.prior_type:
            self.Prior = VampPrior(self.num_prior_components, self.prior_type, prototype_shape=self.prototype_shape ,temperature_scaler=self.prior_init_scaler)
        
        elif "vq" in self.prior_type:
            self.prior_codebook = nn.Parameter(torch.randn(1, self.z_dim, self.num_prior_components), requires_grad=True)
            
            self.prior_codebook.data.uniform_(-1.0 / self.num_prior_components, 1.0 / self.num_prior_components)
        
    def init_vae(self, hparams):
        
        self.vae_input_size = hparams["vae_input_size"]

        self.prior_temperature_scaler = hparams["vae_prior_temperature_scaler"]
        self.prior_init_scaler = hparams["vae_prior_init_scaler"]
        self.enc_layers = hparams["vae_enc_layers"]
        self.dec_layers = self.enc_layers[::-1]
        self.z_dim = hparams["vae_z_dim"]

        self.stochastic_latent = hparams["vae_stochastic_latent"]
        self.vae_dropout_p = hparams["vae_dropout_p"]
        self.vae_decoder_include_sparsity = hparams["vae_decoder_include_sparsity"]
        self.vae_decoder_convolve_output = hparams["vae_decoder_convolve_output"]
        self.vae_decoder_bayesian = hparams["vae_decoder_bayesian"]
        
        vae_activation = get_activation(hparams["vae_activation"])

        self.vae_encoder = simple_mlp_encoder(self.vae_input_size, self.z_dim, self.enc_layers, vae_activation, dropout_p=self.vae_dropout_p,layernorm=self.layernorm,skipconnections=self.skipconnections)
                
        init_all(self.vae_encoder, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["vae_activation"]))
        
        #init_all(self.fc_mu_z, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["vae_latent_activation"]))
        
        self.mu_bias_init = 0.1
        self.log_var_bias_init =  -10.0
        enc_out_dim = self.vae_encoder.model.linear_layers[-1].out_features
        

        self.fc_mu_z = MLP(enc_out_dim, [self.z_dim]*hparams["vae_latent_n_layers"], self.z_dim, get_activation(hparams["vae_latent_activation"]), 
                            layernorm=self.layernorm,skipconnections=self.skipconnections, skiptemperature=self.skiptemperature)
        self.fc_rho_z = MLP(enc_out_dim, [self.z_dim]*hparams["vae_latent_n_layers"], self.z_dim, get_activation(hparams["vae_latent_activation"]), 
                            layernorm=self.layernorm,skipconnections=self.skipconnections, skiptemperature=self.skiptemperature)
        
        init_all(self.fc_mu_z, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["vae_latent_activation"]))
        init_all(self.fc_rho_z, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["vae_latent_activation"]))
        
        # DECODER
        params = dict()
        params['seq_len'] = hparams["L"] if not ("Lmax" in self.model_type) else hparams["variable_length_H"]
        params['alphabet_size'] = hparams["d"]
        params['hidden_layers_sizes'] = self.dec_layers
        params["vae_decoder_bayesian"] = self.vae_decoder_bayesian
        params['z_dim'] = self.z_dim
        params['dropout_proba'] = self.vae_dropout_p
        params['convolve_output'] = self.vae_decoder_convolve_output > 0
        params['convolution_output_depth'] = self.vae_decoder_convolve_output
        params['include_temperature_scaler'] = False #self.model_type == ""
        params['include_sparsity'] = self.vae_decoder_include_sparsity > 0
        params['num_tiles_sparsity'] = self.vae_decoder_include_sparsity
        params["first_hidden_nonlinearity"] = hparams["vae_activation"]
        params["last_hidden_nonlinearity"] = hparams["vae_activation"]

        self.vae_decoder = VAE_Bayesian_MLP_decoder(params)

        if not self.vae_decoder.bayesian_decoder:
            init_all(self.vae_decoder, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["vae_activation"]))

        self.init_latent(hparams)

    def init_transformers(self,hparams):
        self.embed_dim = hparams["transformer_embed_dim"]

        in_dim = self.hparams["transformer_embed_dim"]
        key_dim = hparams["transformer_key_dim"]   #self.dim2//2
        num_heads = hparams["transformer_num_heads"]
        num_layers = hparams["transformer_num_layers"]
        kernel_size = hparams["transformer_kernel_size"]
        use_structure = hparams["transformer_use_structure"]
        
        out_dim = self.hparams["transformer_embed_dim"]  ###self.dim2

        self.encoding_transformer = MultiSelfAttention(
            in_dim, key_dim, out_dim, num_heads, 
            num_layers=num_layers, kernel_size=kernel_size, 
            dropout_p=hparams["transformer_dropout_p"], 
            activation=hparams["transformer_activation"], 
            attn_mask=self.adjacency_matrix if use_structure else None, 
            attnffn_num_layers=hparams["transformer_attnffn_num_layers"]
        )

        init_all(self.encoding_transformer.fullyconnected, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["transformer_activation"]))

        self.decoding_transformer = MultiSelfAttention(
            in_dim, key_dim, out_dim, num_heads, 
            num_layers=num_layers, kernel_size=kernel_size, attn_mask=self.adjacency_matrix if use_structure else None, 
            dropout_p=hparams["transformer_dropout_p"],activation=hparams["transformer_activation"],attnffn_num_layers=hparams["transformer_attnffn_num_layers"]
        )

        init_all(self.decoding_transformer.fullyconnected, nn.init.xavier_uniform_, gain=nn.init.calculate_gain(hparams["transformer_activation"]))

    def forward(self, x, stochastic_latent=True):
        N,L,d = x.shape
        kwarg_stochastic_latent = stochastic_latent

        latent_output = {}

        # Encoder
        if "transformers" in self.model_type:
            xout = self.encoding_transformer(x)
            x = xout
            latent_output["transformers_yout"] = xout#.detach()
        
        if "Lmax" in self.model_type:
            
            x = self.MLP_down(x)
            self.MLP_up.current_L = L
            latent_output["variable_length"] = x#.detach()

        if "mean" in self.model_type:
            xin = x
            x = torch.mean(x,dim=1,keepdim=True)
        
        start_dim, end_dim = x.shape[1:]
        x = torch.flatten(x, start_dim=1, end_dim=2)

        enc_out = self.vae_encoder(x)

        # Latent space
        mu = self.fc_mu_z(enc_out)
        log_var = rho_to_logvar(self.fc_rho_z(enc_out))
        
        if "vq" in self.prior_type:
            embedding_distance = (enc_out.unsqueeze(-1) - self.prior_codebook).pow(2).sum(1)
            closest_embedding_idx = embedding_distance.argmin(-1)
            z_q = self.prior_codebook[...,closest_embedding_idx].transpose(1,2).squeeze(0)
            z = z_q
        else:
            if self.stochastic_latent and kwarg_stochastic_latent:
                z = self.sample_latent(mu, log_var)
            else:
                z = mu
        
        latent_output = {**latent_output, "z": z, "mu":mu,"log_var":log_var}

        #  KL div
        if "mog" in self.prior_type:
            kl = self.Prior.kl_div(mu, log_var)
            latent_output["KLD"] = kl
        
        elif "vamp" in self.prior_type:
            prototypes_in = self.Prior.prepare_prototypes()
            if "transformers" in self.model_type:
                #self.encoding_transformer.requires_grad_(False)
                prototypes_in = self.encoding_transformer(prototypes_in)
                #self.encoding_transformer.requires_grad_(True)

            if "Lmax" in self.model_type:
                #self.MLP_down.requires_grad_(False)
                prototypes_in = self.MLP_down(prototypes_in)
                #self.MLP_down.requires_grad_(True)
            
            prototypes_in = torch.flatten(prototypes_in, start_dim=1, end_dim=2)

            #self.vae_encoder.requires_grad_(False)
            #self.fc_mu_z.requires_grad_(False)
            #self.fc_rho_z.requires_grad_(False)
            prototypes_out = self.vae_encoder(prototypes_in)
            prototypes_mu = self.fc_mu_z(prototypes_out).unsqueeze(0).transpose(1,2)
            prototypes_log_var = rho_to_logvar(self.fc_rho_z(prototypes_out)).unsqueeze(0).transpose(1,2)
            #self.vae_encoder.requires_grad_(True)
            #self.fc_mu_z.requires_grad_(True)
            #self.fc_rho_z.requires_grad_(True)

            kl = self.Prior.kl_div(mu, log_var, prototypes_mu, prototypes_log_var)
            latent_output["KLD"] = kl
        
        elif "vq" in self.prior_type:
            beta = 0.25
            loss = ((z_q.detach() - enc_out)**2 + beta*(z_q - enc_out.detach()) ** 2).sum(1)
            latent_output["KLD"] = loss
        
        if self.vae_decoder.bayesian_decoder:
            latent_output["KLD_params"] = self.vae_decoder.get_KL_div()
        


        ## Predicting DMS
        if self.latent_to_dms_detach:
            y_hat_dms = self.latent_to_dms(enc_out.detach())
        else:
            y_hat_dms = self.latent_to_dms(enc_out)
        
        if self.latent_to_dms_sigmoid:
            y_hat_dms = F.sigmoid(y_hat_dms)

        latent_output["y_hat_dms"]=y_hat_dms


        # Decoder
        logits = self.vae_decoder(z)
        
        logits = torch.unflatten(logits, -1, (start_dim, end_dim))
        if "mean" in self.model_type:
            logits = xin + logits
        if "Lmax" in self.model_type:
            logits = self.MLP_up(logits)
            self.MLP_up.current_L = None

        if "transformers" in self.model_type:
            logits = self.decoding_transformer(logits)
 
        if self.include_temperature_scaler>0:
            #if self.bayesian_decoder:
            #temperature_scaler = self.sampler(self.temperature_scaler_mean, self.temperature_scaler_log_var)
            # else:
            temperature_scaler = self.temperature_scaler_mean
            logits = torch.log(1.0+torch.exp(temperature_scaler)) * logits
            ###xhat = xhat - torch.max(xhat, dim=2, keepdim=True).values
        
        return logits, latent_output

    def sample_latent(self, mu, log_var):
        """
            Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu, device=mu.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z

