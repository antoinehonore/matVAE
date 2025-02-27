import torch
import torch.nn.functional as F

import matplotlib
import socket
import matplotlib.pyplot as plt
if socket.gethostname()=="cmm0958":
    matplotlib.use('tkagg') 
else:
    matplotlib.use('agg') 

import pandas as pd

import lightning as L

from torcheval.metrics.functional import r2_score, binary_auprc, binary_auroc, binary_precision, binary_recall, binary_f1_score, binary_accuracy

from torchmetrics.functional.regression.spearman import _spearman_corrcoef_compute
import sys


import torchsort

class minEVE(torch.nn.Module):
    def __init__(self, hparams):
        super(minEVE, self).__init__()

        self.vae_input_size = hparams["vae_input_size"]
        self.dim1 = hparams["dim1"]
        self.dim2 = hparams["dim2"]
        self.prior_temperature_scaler = hparams["vae_prior_temperature_scaler"]
        self.prior_init_scaler = hparams["vae_prior_init_scaler"]
        self.enc_layers = hparams["vae_enc_layers"]
        self.dec_layers = self.enc_layers[::-1]
        self.z_dim = hparams["vae_z_dim"]
        self.num_prior_components = hparams["vae_num_prior_components"]
        self.prior_type = hparams["vae_prior_type"]
        self.Lmax = hparams["variable_length_Lmax"]
        self.H = hparams["variable_length_H"]
        self.model_type = hparams["model_type"]

        self.stochastic_latent = hparams["vae_stochastic_latent"]
        self.vae_dropout_p = hparams["vae_dropout_p"]
        self.output_scaler = hparams["output_scaler"]
        self.reconstruction_error = hparams["reconstruction_error"]
        self.vae_decoder_include_sparsity = hparams["vae_decoder_include_sparsity"]
        self.vae_decoder_convolve_output = hparams["vae_decoder_convolve_output"]
        self.vae_decoder_bayesian =hparams["vae_decoder_bayesian"]
        if "vae_activation" in hparams.keys():
            if hparams["vae_activation"] == "leakyrelu":
                activation = torch.nn.LeakyReLU
            elif hparams["vae_activation"] == "relu":
                activation = torch.nn.ReLU
            elif hparams["vae_activation"] == "gelu":
                activation = torch.nn.GELU
            elif hparams["vae_activation"] == "relu^2":
                activation = torch.nn.GELU
            else:
                print("Unknown", hparams["activation"])
                sys.exit(1)

        self.output_map = torch.nn.Identity()
        if "output_function" in hparams.keys():
            if hparams["output_function"] == "tanh":
                self.output_map = torch.nn.Tanh()
            elif hparams["output_function"] == "identity":
                self.output_map = torch.nn.Identity()
            elif hparams["output_function"] == "clamp":
                self.output_map = partial(torch.clamp, min=-1, max=1)
            elif hparams["output_function"] == "sigmoid":
                self.output_map = torch.sigmoid
            else:
                print("Unknown", hparams["output_function"])
                sys.exit(1)
        
        if "transformers" in self.model_type:
            self.embed_dim = hparams["transformer_embed_dim"]
            self.vae_input_size = self.embed_dim * self.dim1

            in_dim = self.embed_dim
            key_dim = hparams["transformer_key_dim"]   #self.dim2//2
            num_heads = hparams["transformer_num_heads"]
            num_layers = hparams["transformer_num_layers"]
            kernel_size = hparams["transformer_kernel_size"]
            self.truncate_num = hparams["transformer_truncate"] if hparams["transformer_truncate"] > 0 else None
            out_dim = self.embed_dim  ##self.dim2

            self.encoding_transformer = MultiSelfAttention(
                in_dim, key_dim, out_dim, num_heads, 
                num_layers=num_layers, kernel_size=kernel_size, 
                dropout_p=hparams["transformer_dropout_p"],activation=hparams["transformer_activation"], attnffn_num_layers=hparams["transformer_attnffn_num_layers"]
            )
            self.decoding_transformer = MultiSelfAttention(
                in_dim, key_dim, out_dim, num_heads, 
                num_layers=num_layers, kernel_size=kernel_size, 
                dropout_p=hparams["transformer_dropout_p"],activation=hparams["transformer_activation"],attnffn_num_layers=hparams["transformer_attnffn_num_layers"]
            )

        self.vae_encoder = simple_mlp_encoder(self.vae_input_size, self.z_dim, self.enc_layers, activation, dropout_p=self.vae_dropout_p)
        
        self.mu_bias_init = 0.1
        self.log_var_bias_init = -10.0
        
        self.fc_mu_z = torch.nn.Linear(self.z_dim, self.z_dim)
        self.fc_rho_z = torch.nn.Linear(self.z_dim, self.z_dim)

        nn.init.constant_(self.fc_mu_z.bias, self.mu_bias_init)
        nn.init.constant_(self.fc_rho_z.weight, 0) ## self.log_var_bias_init)
        nn.init.constant_(self.fc_rho_z.bias, 0)   ## self.log_var_bias_init)


        #if hparams["vae_decoder"] == "simple":
        #    self.vae_decoder = simple_mlp_decoder(self.vae_input_size,self.z_dim, self.dec_layers, activation, dropout_p=self.dropout_p)

        #elif hparams["vae_decoder"] == "bayesian":
        params = dict()
        params['seq_len'] = hparams["dim1"]
        params['alphabet_size'] = hparams["dim2"]
        params['hidden_layers_sizes'] = self.dec_layers
        params["vae_decoder_bayesian"] = self.vae_decoder_bayesian
        params['z_dim'] = self.z_dim
        params['dropout_proba'] = self.vae_dropout_p
        params['convolve_output'] = self.vae_decoder_convolve_output > 0
        params['convolution_output_depth'] = self.vae_decoder_convolve_output
        params['include_temperature_scaler'] = True
        params['include_sparsity'] = self.vae_decoder_include_sparsity > 0
        params['num_tiles_sparsity'] = self.vae_decoder_include_sparsity
        params["first_hidden_nonlinearity"] = "relu"
        params["last_hidden_nonlinearity"] = "relu"
        self.vae_decoder = VAE_Bayesian_MLP_decoder(params)

        self.log_var = nn.Parameter(torch.log(torch.tensor(0.01)),requires_grad=False)

        if "mog" in self.prior_type:
                if self.num_prior_components==1:
                    self.prior_means =  nn.Parameter(self.prior_init_scaler * torch.zeros(1, self.z_dim, self.num_prior_components, requires_grad=False))
                else:
                    self.prior_means =  nn.Parameter(self.prior_init_scaler * torch.randn(1, self.z_dim, self.num_prior_components, requires_grad=True))
            
        
        if self.prior_type == "mog":
            self.prior_rhos = nn.Parameter(
                torch.randn(1, self.z_dim, self.num_prior_components)
            )
        
        elif self.prior_type == "mog_fixvar":
            self.prior_rhos = nn.Parameter((math.log(math.e - 1)) * torch.ones(1, self.z_dim, self.num_prior_components), requires_grad=False)
        
        if "vamp" in self.prior_type:
            self.prototypes = nn.Parameter(
                torch.randn(self.num_prior_components, self.dim1, self.dim2)
            )

        if "Lmax" in self.model_type:
            n_layers = hparams["variable_length_n_layers"]
            
            variable_length_layers = [i / (n_layers) for i in range(1,n_layers)][::-1]

            self.MLP_down = VarLenMLPdown(self.Lmax, self.H, variable_length_layers)
            self.MLP_up = VarLenMLPup(self.Lmax, self.H, variable_length_layers[::-1])
        self.internal_metrics={}
        
    def forward(self, y):
        y = y.to(torch.float)
        if "transformers" in self.model_type:
            N,L,d = y.shape
            yout = self.encoding_transformer(y)
            y = yout
            
            if "qr" in self.model_type:
                Q, y = torch.linalg.qr(yout, mode="reduced")
            
            elif "Lmax" in self.model_type:
                y = self.MLP_down(y)
                self.MLP_up.current_L = L

            y = torch.flatten(y, start_dim=1, end_dim=2)

        enc_out = self.vae_encoder(y)

        # Default in case self.stochastic_latent == False
        z = enc_out
        mu = z
        log_var = self.log_var
        
        if self.stochastic_latent and (self.training):
            mu = self.fc_mu_z(enc_out)
            log_var = rho_to_logvar(self.fc_rho_z(enc_out))
            z = self.sample_latent(mu, log_var)

        yhat = self.vae_decoder(z)
        
        if "transformers" in self.model_type:
            yhat = torch.unflatten(yhat, -1, (self.dim1, self.dim2))
            
            if "qr" in self.model_type:
                yhat = Q.detach() @ yhat

            elif "Lmax" in self.model_type:
                yhat = self.MLP_up(yhat)
                self.MLP_up.current_L = None

            else:
                yhat = self.decoding_transformer(yhat)
        
        yhat = self.output_map(yhat)

        # mu and log_var are needed because they are plotted
        latent_output = {"mu":mu, "log_var":log_var}
        
        if self.prior_type == "default":
            latent_output["KLD"] = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(1)
        
        elif "mog" in self.prior_type:
            normalizing_constant = math.log(self.num_prior_components)
            kl_divs = kl_divergence_two_gaussians(mu.unsqueeze(-1), log_var.unsqueeze(-1), self.prior_means, rho_to_logvar(self.prior_rhos))
            kl = normalizing_constant - torch.logsumexp(- kl_divs, dim=1)
            latent_output["KLD"] = kl
        
        elif "vamp" in self.prior_type:
            if (self.dim1 != self.dim2) and (not ("Lmax" in self.model_type)): 
                # This happens when neither QR nor SVD are used. 
                # FIXME: the condition can lead to issues when L == d and the default input data is used
                if "gumbel" in self.prior_type:
                    prototypes = F.gumbel_softmax(self.prototypes, tau=self.prior_temperature_scaler, hard=True)
                else:
                    prototypes = torch.softmax(self.prior_temperature_scaler * self.prototypes, 2)
            else:
                # When the data are encoded via SVD or QR or transformers, do not modify the prototypes
                prototypes = self.prototypes
            
            proto_view = self.prototypes.view(self.prototypes.shape[0], -1)
            
            self.internal_metrics = {   **{"prototypes/mean/"+str(i):m.item() for i,m in enumerate(proto_view.mean(1))},
                                        **{"prototypes/std/"+str(i):m.item() for i,m in enumerate(proto_view.std(1))}
                                        }
            was_training_mode = self.vae_encoder.training
            #if was_training_mode:
            #    self.vae_encoder.eval()
            
            prototypes_out = self.vae_encoder(torch.flatten(prototypes, start_dim=1, end_dim=2))
            
            #if was_training_mode:
            #    self.vae_encoder.train()
            
            prototypes_mu = self.fc_mu_z(prototypes_out).unsqueeze(0).transpose(1,2)
            prototypes_log_var = rho_to_logvar(self.fc_rho_z(prototypes_out)).unsqueeze(0).transpose(1,2)
            normalizing_constant = math.log(self.num_prior_components)
            kl_divs = kl_divergence_two_gaussians(mu.unsqueeze(-1), log_var.unsqueeze(-1), prototypes_mu, prototypes_log_var)
            kl = normalizing_constant - torch.logsumexp(- kl_divs, dim=1)
            latent_output["KLD"] = kl

        return yhat, latent_output

    def sample_latent(self, mu, log_var):
        """
            Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu, device=mu.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z

# trainer class
class lightningEVE(L.LightningModule):
    def __init__(self, model, hparams=None):
        super(lightningEVE, self).__init__()
        self.model =              model
        training_parameters =     hparams["training_parameters"]
        self.training_parameters = training_parameters
        self.loss_fun =           self.training_parameters["loss_fun"]
        
        self.x_recon_figure     = plt.subplots(1,2, figsize=(10,6))
        self.y_recon_figure     = plt.subplots(figsize=(10,6))
        
        self.val_x_recon_figure = plt.subplots(1,2, figsize=(10,6))
        self.val_y_recon_figure = plt.subplots(figsize=(10,6))
        self.val_elbo_figure =    plt.subplots(figsize=(10,6))
        self.val_score_figure =   plt.subplots(1, 2, figsize=(14, 6))

        self.plot_batch_index   = -1
        self.plot_elbos = False
        self.latest_cat_loss    = 0
        self.the_training_step  = 0
        self.best_val_mse       = torch.inf

        self.save_hyperparameters(hparams)
        self.val_scores = {}
        self.val_scores_msa = {}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/x_mse/val": torch.nan, "hp/x_mse/train": torch.nan,"hp/max_grad": torch.nan})

        self.val_scores = {} # {k: [] for k in self.trainer.val_dataloaders_info["name"] if k.startswith("dms")}
        self.val_scores_msa = {}
    def cat_loss_func(self, logits, x, cat_loss_name="CE"):
        """Input of size (N,L,d), output of size (N,L). Computes the cross-entropy between predictions and x. """
        ##  return F.binary_cross_entropy_with_logits(xhat, x, reduction='none').sum(2)
        
        if cat_loss_name == "CE":
            """In this case prediction should be unnormalized"""
            return F.cross_entropy(logits.transpose(1,2), x.transpose(1,2), reduction="none")#nll_loss(log_xhat.transpose(1,2), torch.argmax(x, dim=-1), reduction='none')
        
        elif cat_loss_name == "BCE": # similar to original: mistke of using log softmax twice
            log_xhat = torch.log_softmax(logits, dim=-1)
            return F.binary_cross_entropy_with_logits(log_xhat, x, reduction='none').sum(2)

    def training_step_msa(self,batch,batch_idx):
        #if dataloader_idx == 0:
        #batch_weight = 1/self.trainer.train_dataloaders_info["n_batches"][dataloader_idx]
        #for protein_name, batch in batch_dict.items():

        x, yin, y, y_flat = self.get_input_data(batch)
        #x_reverse, yin_reverse, y_reverse, y_flat_reverse = self.get_input_data(batch,reverse=True)

        yhat, latent_output = self.model.forward(yin)
        #yhat_reverse, latent_output_reverse = self.model.forward(yin_reverse)

        neg_ELBO_leftright, KLD_latent_leftright, CE_dict_leftright, mse_dict_leftright = self.my_loss_function(yhat, y_flat, latent_output, x=x, batch=batch, reconstruction_error=self.model.reconstruction_error)
        #neg_ELBO_reverse, KLD_latent_reverse, CE_dict_reverse, mse_dict_reverse = self.my_loss_function(yhat_reverse, y_flat_reverse, latent_output_reverse, x=x_reverse, batch=batch, reconstruction_error=self.model.reconstruction_error)
        
        neg_ELBO = neg_ELBO_leftright#(neg_ELBO_leftright + neg_ELBO_reverse) / 2
        KLD_latent = KLD_latent_leftright#(KLD_latent_leftright + KLD_latent_reverse)/2
        mse_dict = {}
        CE_dict = {}
        
        mse_dict["x"] = mse_dict_leftright["x"]# (mse_dict_leftright["x"]+mse_dict_reverse["x"])/2
        mse_dict["y"] = mse_dict_leftright["y"]#(mse_dict_leftright["y"]+mse_dict_reverse["y"])/2
        CE_dict["x"] = CE_dict_leftright["x"]# (CE_dict_leftright["x"]+CE_dict_reverse["x"])/2
        CE_dict["y"] = CE_dict_leftright["y"]#(CE_dict_leftright["y"]+CE_dict_reverse["y"])/2

        if (batch_idx == self.plot_batch_index):
            isample=0
            axes = self.x_recon_figure[1]
            axes[1].cla()
            axes[0].cla()
            log_recon = torch.log_softmax(self.get_reconstruction_data(yhat, batch),dim=-1)

            axes[1].imshow(log_recon[isample][:37].detach().exp().cpu(), aspect="auto",vmin=0,vmax=1)
            axes[0].imshow(x[isample][:37].detach().cpu(), aspect="auto",vmin=0,vmax=1)
            self.logger.experiment.add_figure("X Reconstruction/Train", self.x_recon_figure[0], self.the_training_step)
            
            axes = self.y_recon_figure[1]
            axes.cla()
            ### axes[0].cla()
            
            axes.plot(y_[isample].detach().cpu(), label="y",color="gray", lw=2,alpha=0.6)
            axes.plot(yhat[isample].detach().cpu(),label="yhat",ls="--",color="darkred")
            self.logger.experiment.add_figure("Y Reconstruction/Train", self.y_recon_figure[0], self.the_training_step)
        
        log_dict = {
                    "train_neg_ELBO": neg_ELBO.mean(), 
                    "train_x_CE": CE_dict["x"].mean(), "train_y_CE": CE_dict["y"].mean(),
                    "train_KLD_latent": KLD_latent.mean(), 
                    "train_x_mse": mse_dict["x"].mean(), "train_y_mse": mse_dict["y"].mean(), 
                    "hp/x_mse/train":mse_dict["x"].mean(), #**self.model.internal_metrics
                    # "train_log_var": latent_output["log_var"].mean(),
                    # "train_mu": latent_output["mu"].mean()
                }
        
        self.log_dict(log_dict, on_epoch=True, on_step=False, batch_size=neg_ELBO.shape[0])
        #self.log_dict(self.model.internal_metrics)

        self.the_training_step += 1
        loss = neg_ELBO.mean()
        return loss
    

    def training_step(self, batch_input, batch_idx, dataloader_idx=0):
        loss = self.training_step_msa(batch_input, batch_idx)
        return loss


    def get_input_data(self, batch, reverse=False):
        x = batch["x"]

        if not ("mat_reconstruction" in batch.keys()): 
            y_matrix = batch["x"]
        else:

            y_matrix = batch["y"]

        if reverse:
            x = torch.flip(x,dims=(1,))
            y_matrix = torch.flip(y_matrix,dims=(1,))
        
        # Flatten
        y_flattened = y_matrix.view(y_matrix.shape[0], y_matrix.shape[1] * y_matrix.shape[2])
        if "transformers" in self.model.model_type:
            yin = y_matrix
        else:
            yin = y_flattened
        return x, yin, y_matrix, y_flattened

    def get_reconstruction_data(self, yhat, batch):
        """Project/reconstruct yhat back to input space with reconstruction transform, remove positional encoding"""
        
        x, yin, y, y_flat = self.get_input_data(batch)
        
        if "mat_reconstruction" in batch.keys():
            _yhat = yhat.view(y.shape[0], y.shape[1], y.shape[2])
            
            logits = batch["mat_reconstruction"] @ _yhat
            
            npos_encoding = logits.shape[-1] - x.shape[-1]
            
            logits = logits[...,npos_encoding:]
            
            if "pos_encoding" in batch.keys():
                logits = logits - batch["pos_encoding"]

            logits = self.model.output_scaler*logits
            
        else:
            logits = yhat.view(y.shape[0], y.shape[1], y.shape[2])

        return logits

    def log_p_approx(self, batch):
        x, yin, y, y_flat = self.get_input_data(batch)
        #x_reverse, yin_reverse, y_reverse, y_flat_reverse = self.get_input_data(batch,reverse=True)

        yhat, latent_output = self.model.forward(yin)
        #yhat_reverse, latent_output_reverse = self.model.forward(yin_reverse)

        neg_ELBO, _, _, _ = self.my_loss_function(yhat, y_flat, latent_output, batch=batch, reduction="none",  x=x, reconstruction_error=self.model.reconstruction_error)
        #neg_ELBO_reverse, _, _, _ = self.my_loss_function(yhat_reverse, y_flat_reverse, latent_output_reverse, batch=batch, reduction="none",  x=x_reverse, reconstruction_error=self.model.reconstruction_error)

        return - neg_ELBO#(neg_ELBO + neg_ELBO_reverse)/2

    def my_loss_function(self, yhat, y, latent_output, reduction="mean", x=None, batch=None, reconstruction_error="CE"):
        """"""
        logits = self.get_reconstruction_data(yhat, batch)

        # KLD between Normal and N(z_mu, z_sigma)
        KLD_latent = latent_output["KLD"]#
        
        cat_loss_name = "BCE" if ("BCE" in reconstruction_error) else "CE"
        x_CE = self.cat_loss_func(logits, x.type_as(logits), cat_loss_name=cat_loss_name).mean(1)

        reconstruct_loss = 0
        if "x_CE" in reconstruction_error:
            reconstruct_loss += x_CE
        
        if "y_CE" in reconstruction_error:
            # y must be a projected
            assert("mat_reconstruction" in batch.keys())
            # Rescale target between 0,1, assume that yhat is between 0,1

            y_CE = F.binary_cross_entropy((yhat + 1)/2, (y.type_as(yhat) + 1)/2, reduction="none").mean(1)
            reconstruct_loss += y_CE
        else:
            y_CE = x_CE


        y_mse = F.mse_loss(yhat.reshape(*y.shape), y, reduction="none").mean(1)

        log_xhat = torch.log_softmax(logits, dim=-1)
        
        x_mse = F.mse_loss(log_xhat.exp(), x.type_as(log_xhat), reduction="none").mean(-1).mean(1)

        if "y_mse" in reconstruction_error:
            reconstruct_loss += y_mse
        
        if "x_mse" in reconstruction_error:
            reconstruct_loss += x_mse

        neg_ELBO = reconstruct_loss + self.training_parameters['kl_latent_scale']*KLD_latent

        return neg_ELBO, KLD_latent, {"x": x_CE, "y": y_CE}, {"x": x_mse, "y": y_mse}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        dataloader_name = self.trainer.val_dataloaders_info["name"][dataloader_idx]
        if not (dataloader_name.startswith("dms")):
            x, yin, y, y_flat = self.get_input_data(batch)

            #x_reverse, yin_reverse, y_reverse, y_flat_reverse = self.get_input_data(batch,reverse=True)

            yhat, latent_output = self.model.forward(yin)
            #yhat_reverse, latent_output_reverse = self.model.forward(yin_reverse)

            neg_ELBO_normal, KLD_latent_normal, CE_dict_normal, mse_dict_normal = self.my_loss_function(yhat, y_flat, latent_output, x=x, batch=batch, reconstruction_error=self.model.reconstruction_error)
            #neg_ELBO_reverse, KLD_latent_reverse, CE_dict_reverse, mse_dict_reverse = self.my_loss_function(yhat_reverse, y_flat_reverse, latent_output_reverse, x=x_reverse, batch=batch, reconstruction_error=self.model.reconstruction_error)
            
            neg_ELBO = neg_ELBO_normal#(neg_ELBO_normal + neg_ELBO_reverse) / 2
            KLD_latent =KLD_latent_normal# (KLD_latent_normal + KLD_latent_reverse)/2
            mse_dict = {}
            CE_dict = {}
            
            mse_dict["x"] = mse_dict_normal["x"]#(mse_dict_normal["x"]+mse_dict_reverse["x"])/2
            mse_dict["y"] = mse_dict_normal["y"]#(mse_dict_normal["y"]+mse_dict_reverse["y"])/2
            CE_dict["x"] = CE_dict_normal["x"]#(CE_dict_normal["x"]+CE_dict_reverse["x"])/2
            CE_dict["y"] =CE_dict_normal["y"]# (CE_dict_normal["y"]+CE_dict_reverse["y"])/2

            log_dict = {"val_neg_ELBO": neg_ELBO.mean(),  
                        "val_x_mse": mse_dict["x"].mean(), "val_y_mse": mse_dict["y"].mean(), 
                        "val_KLD_latent": KLD_latent.mean(),  
                        "val_x_CE": CE_dict["x"].mean(),"val_y_CE": CE_dict["y"].mean(),
                       # "val_log_var": latent_output["log_var"].mean(), "val_mu": latent_output["mu"].mean()
                        }
            
            if log_dict["val_x_mse"] < self.best_val_mse:
                new_best = log_dict["val_x_mse"]
            else:
                new_best = self.best_val_mse
            
            self.best_val_mse = new_best
            
            log_dict = {**log_dict, "hp/x_mse/val": self.best_val_mse}
            self.val_scores_msa[dataloader_name] = log_dict

            #self.log_dict(log_dict, add_dataloader_idx=True)
            
            if (batch_idx == self.plot_batch_index):
                isample = 0
                axes = self.val_x_recon_figure[1]
                axes[1].cla()
                axes[0].cla()
                log_recon = torch.log_softmax(self.get_reconstruction_data(yhat, batch),dim=-1)
                axes[1].imshow(log_recon[isample][:37].detach().exp().cpu(), aspect="auto",vmin=0,vmax=1)
                axes[0].imshow(x[isample][:37].detach().cpu(), aspect="auto",vmin=0,vmax=1)
                self.logger.experiment.add_figure("X Reconstruction/Val", self.val_x_recon_figure[0], self.the_training_step)
                
                axes = self.val_y_recon_figure[1]
                axes.cla()
                
                axes.plot(y_[isample].detach().cpu(), label="y",color="gray", lw=2,alpha=0.6)
                axes.plot(yhat[isample].detach().cpu(),label="yhat",ls="--",color="darkred")
                self.logger.experiment.add_figure("Y Reconstruction/Val", self.val_y_recon_figure[0], self.the_training_step)

        # If we are dealing with dataloaders of DMS data
        else:
            if not ("wt_mat_reconstruction" in batch.keys()):
                wt_batch = {"x": batch["x_wt"]}
            else:
                wt_batch = {"x": batch["x_wt"], "mat_reconstruction":batch["wt_mat_reconstruction"], "y":batch["y_wt"]} 
            
            if not ("mat_reconstruction" in batch.keys()):
                data_batch = {  "y": batch["x"], 
                                "x": batch["x"]}
            else:
                data_batch = {"x":batch["x"], "mat_reconstruction": batch["mat_reconstruction"], "y": batch["y"]}
            

            if "pos_encoding" in batch.keys():
                wt_batch = {**wt_batch , "pos_encoding":batch["wt_pos_encoding"]}
                data_batch = {**data_batch , "pos_encoding":batch["pos_encoding"]}

            elbo = self.log_p_approx(data_batch)
            ELBO_WT = self.log_p_approx(wt_batch)
            
            ypred = elbo - ELBO_WT
            
            if self.plot_elbos: # Skip plotting this for efficiency
                ax = self.val_elbo_figure[1]
                ax.cla()
                ax.plot(ELBO_WT.cpu(), label="WT ELBO", lw=2, color="black")
                ax.plot(elbo.cpu(), label="V ELBO", color="darkred", ls="--")
                ax.legend()
                self.logger.experiment.add_figure("WT_V_ELBO", self.val_elbo_figure[0], self.the_training_step)
            if not (dataloader_name in self.val_scores.keys()):
                self.val_scores[dataloader_name] = []
            self.val_scores[dataloader_name].append([ypred[:, None], batch["target"][:, None],batch["target_bin"][:, None], elbo[:, None],ELBO_WT[:, None]])

    def on_validation_epoch_end(self):
        
        if all([len(v)>0 for v in self.val_scores.values()]):
            for dataloader_name, protein_val_scores in self.val_scores.items():
                dataloader_index = self.trainer.val_dataloaders_info["name"].index(dataloader_name)
                #T = self.trainer.val_dataloaders_info["T"][dataloader_index]
                # Concatenate and plot the DMS scores
                all_dms_output = torch.cat(list(map(lambda l: torch.cat(l[:3], dim=1), protein_val_scores))).detach().cpu()
                
                all_elbo_output = torch.cat(list(map(lambda l: torch.cat(l[2:], dim=1),protein_val_scores))).detach().cpu()

                predictions = all_dms_output[:, 0]#.detach().cpu()
                targets = all_dms_output[:, 1]#.detach().cpu()
                targets_bin = all_dms_output[:, 2]#.detach().cpu()
                thescores = get_dms_scores(predictions, targets, targets_bin)
                log_dict = {"{}/{}".format(score_name, dataloader_name): score_value for score_name,score_value in thescores.items()}
                
                if False:
                    axes = self.val_score_figure[1]
                    axes[0].cla()
                    axes[0].scatter(targets, predictions)
                    axes[0].set_xlabel("DMS Targets")
                    axes[0].set_ylabel("Predictions")

                    axes[1].cla()
                    axes[1].scatter(all_elbo_output[:,0], all_elbo_output[:,1])
                    axes[1].set_xlabel("ELBO")
                    axes[1].set_ylabel("ELBO WT")

                    self.logger.experiment.add_figure("DMS Target-predictions ({})".format(dataloader_name), 
                        self.val_score_figure[0], self.the_training_step
                        )
                
                self.log_dict(log_dict)
            self.val_scores = {k:[] for k in self.trainer.val_dataloaders_info["name"] if k.startswith("dms")}
        
        # Move all the scores to cpu
        self.val_scores_msa = {k1:{k2:v.item() if not (isinstance(v,float)) else v for k2,v in self.val_scores_msa[k1].items()} for k1 in self.val_scores_msa.keys()}
        log_dict = pd.DataFrame(self.val_scores_msa).mean(1).to_dict()
        
        self.log_dict(log_dict)
        #self.val_scores_msa = {k:[] for k in self.trainer.val_dataloaders_info["name"] if not k.startswith("dms")}

    def on_before_optimizer_step(self,optimizer):
        if any([torch.isnan(p).any() for p in self.parameters()]):
            self.log_dict({"hp/max_grad": torch.nan}, on_step=True)
            sys.exit(0)
        
        max_grad = max([param.grad.abs().max() for param in self.parameters() if not (param.grad is None)])
        #grad_var = [param.grad.abs().max() for param in self.parameters() if not (param.grad is None)]
        self.log_dict({"hp/max_grad": max_grad}, on_step=True)
        if (self.the_training_step>10) & (self.training_parameters['gradient_clip_norm']>0):
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.training_parameters['gradient_clip_norm'])

        for param in self.parameters():
            if not (param.grad is None):# param.requires_grad:
                noise_std = self.training_parameters['gradient_noise_std']#param.grad.std() * 10**(-self.training_parameters['gradient_noise_db']/20)
                param.grad += torch.rand_like(param.grad) * noise_std

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.training_parameters['learning_rate'], weight_decay = self.training_parameters['l2_regularization'])
        if self.training_parameters['use_lr_scheduler']:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=self.training_parameters['lr_scheduler_step_size'], gamma=self.training_parameters['lr_scheduler_gamma'])

            return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return optim
