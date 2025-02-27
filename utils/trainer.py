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
import math



def get_info(theloader_dict):
    return {"name":list(theloader_dict.keys()), "n_batches":[len(loader) for loader in theloader_dict.values()],
                            #"T": [loader.dataset.x.shape[1] for loader in theloader_dict.values()]
                            }

# trainer class
class lightningdmsEVE(L.LightningModule):
    def __init__(self, model=None, hparams=None):
        super(lightningdmsEVE, self).__init__()
        self.model =              model
        self.training_parameters = hparams["training_parameters"]
        self.loss_fun =           self.training_parameters["loss_fun"]
        

        self.x_recon_figure     = plt.subplots(1,2, figsize=(10,6))
        self.y_recon_figure     = plt.subplots(figsize=(10,6))
        self.val_x_recon_figure = plt.subplots(1,2, figsize=(10,6))
        self.val_y_recon_figure = plt.subplots(figsize=(10,6))
        self.val_elbo_figure =    plt.subplots(figsize=(10,6))
        self.val_score_figure =   plt.subplots(1,2,figsize=(10, 6))
        self.val_boxplot_figure =   plt.subplots()


        self.plot_batch_index   = self.training_parameters["plot_batch_index"]
        self.plot_elbos = self.training_parameters["plot_elbos"]
        self.latest_cat_loss    = 0
        self.the_training_step  = 0
        self.best_val_mse       = torch.inf

        self.save_hyperparameters(hparams)
        self.val_scores = {}
        self.val_scores_msa = {}
        self.val_scores_dms = {}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
        {"hp/x_mse/val": torch.nan, "hp/x_mse/train": torch.nan,"hp/max_grad": torch.nan,"hp/earlystop_metric": torch.nan, "val_neg_ELBO":torch.nan})

        self.val_scores = {} 
        self.val_scores_msa = {}
        self.val_scores_dms = {}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        
        loss = 0.0
        log_dict = {}
        use_dms_only = self.training_parameters["use_dms_only"]

        self.the_training_step += 1
        
        # Determine whether DMS data should be used during this step.
        use_dms = False

        if self.training_parameters["use_dms"] == "everyother":
            use_dms = ((self.the_training_step%2)==0) and ("dms" in batch.keys())

        elif isinstance(self.training_parameters["use_dms"], float) or isinstance(self.training_parameters["use_dms"], int):
            use_dms = (self.the_training_step >= self.training_parameters["num_training_steps"]*self.training_parameters["use_dms"]) and ("dms" in batch.keys())
        
        if use_dms:
            x_dms = self.get_input_data(batch["dms"])
            N = x_dms.shape[0]
            score_str = self.training_parameters["latent_to_dms_target"]
            
            y_true = torch.concat([batch["dms"][ss].to(torch.float).unsqueeze(-1) for ss in score_str.split("+")],dim=-1)
            
            # Compute loss
            loss_dms,_ = self.model.compute_loss(x_dms, y=y_true, stochastic_latent=False)
            
            log_dict = { **log_dict,
                        "train_loss_dms": loss_dms
                    }

            loss += loss_dms*self.training_parameters["use_dms_factor"]
        
        use_msa = (not ("dms" in batch.keys())) or (not use_dms) or (("msa" in batch.keys()) and (not use_dms_only))

        # Use of MSA data, if there are no DMS data, or (there is msa and DMS data,  and we should not use only the DMS data )
        if use_msa:
            x = self.get_input_data(batch["msa"])
            N = x.shape[0]
            
            warm_up_scale = annealing_factor(self.training_parameters["annealing_warm_up"], self.the_training_step)
            beta = self.training_parameters["kl_latent_scale"]    ###(self.the_training_step+1)/self.training_parameters["num_training_steps"]
            neg_ELBO, latent_output, x_CE = self.model.compute_loss(x, beta=beta, kl_global_params_scale=warm_up_scale*self.training_parameters['kl_global_params_scale'])

            loss_msa = neg_ELBO.mean()
            
            log_dict = { **log_dict,"train_neg_ELBO": neg_ELBO.mean(), 
                        "train_x_CE": x_CE.mean(),  ### "train_y_CE": CE_dict["y"].mean(),
                        "train_KLD_latent": latent_output["KLD"].mean(), 
                        "train_mu": latent_output["mu"].mean(),
                        "train_logvar": latent_output["log_var"].mean()
                        }
            loss += loss_msa

        self.log_dict(log_dict, on_epoch=True, on_step=False, batch_size=N)
        return loss
    
    def get_input_data(self, batch, reverse=False):
        x = batch["x"]

        if reverse:
            x = torch.flip(x,dims=(1,))

        return x.to(torch.float)
        
    def on_validation_epoch_start(self):
        self.trainer.val_dataloaders_info = get_info(self.trainer.val_dataloaders)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        dataloader_name = self.trainer.val_dataloaders_info["name"][dataloader_idx]
        if not (dataloader_name.startswith("dms")):
            x = self.get_input_data(batch)
            neg_ELBO, latent_output, x_CE = self.model.compute_loss(x,stochastic_latent=False, kl_global_params_scale=0, beta=1)

            log_dict = {"val_neg_ELBO": neg_ELBO.mean(),  
                        "val_KLD_latent": latent_output["KLD"].mean(),  
                        "val_x_CE": x_CE.mean(),
                        "val_mu": latent_output["mu"].mean(),
                        }
            
            self.val_scores_msa[dataloader_name] = log_dict

            if batch_idx == -1:
                self.val_boxplot_figure[1].cla()
                self.val_boxplot_figure[1].boxplot(latent_output["mu"].T.cpu().tolist())
                self.val_boxplot_figure[1].set_title("$\\tau={:.04f}$".format(self.model.matVAE.tau.item()))
                self.val_boxplot_figure[1].set_xlabel("Category")
                self.val_boxplot_figure[1].set_ylabel("Category probability")
                self.val_boxplot_figure[1].set_ylim([0,1])
                self.logger.experiment.add_figure("z_boxplot/val/{}".format(dataloader_name), self.val_boxplot_figure[0], self.the_training_step)

            if (batch_idx == self.plot_batch_index):
                xhat, latent_output = self.model.forward(x, stochastic_latent=False)
                isample = 0
                axes = self.val_x_recon_figure[1]
                axes[1].cla()
                axes[0].cla()
                log_recon = torch.log_softmax(xhat,dim=-1)
                if ("transformers_yout" in latent_output.keys()):
                    axes[1].imshow(latent_output["transformers_yout"][isample].detach().exp().cpu(), aspect="auto")
                else:
                    axes[1].imshow(log_recon[isample].detach().exp().cpu(), aspect="auto", vmin=0, vmax=1)
                axes[0].scatter(range(latent_output["z"].shape[0]),latent_output["z"][:,0].detach().cpu())
                axes[0].set_xlabel("Sample")
                axes[0].set_ylabel("Component 0")

                self.logger.experiment.add_figure("z_comp0-trans_x_im/{}".format(dataloader_name), self.val_x_recon_figure[0], self.the_training_step)
        
        else:        # If dataloaders of DMS data
            x_wt = batch["x_wt"].to(torch.float)
            x = batch["x"].to(torch.float)
            score_str = self.training_parameters["latent_to_dms_target"]
            
            y_true_dms = torch.concat([batch[ss].to(torch.float).unsqueeze(-1) for ss in score_str.split("+")],dim=-1)
            #y_true_dms = batch[self.training_parameters["latent_to_dms_target"]]
            
            neg_ELBO, latent_output, _ = self.model.compute_loss(x, stochastic_latent=False, kl_global_params_scale=0, beta=1)

            # neg_ELBO, KLD_latent, x_CE = self.loss_function(xhat, x, latent_output,reconstruction_error=self.model.reconstruction_error)
            elbo, y_hat_dms = -neg_ELBO, latent_output["y_hat_dms"]
            
            # Forward pass with WT
            #xhat_wt, latent_output_wt = self.model.forward(x_wt, stochastic_latent=False)
            #neg_ELBO_wt, KLD_latent_wt, x_CE = self.loss_function(xhat_wt, x_wt, latent_output_wt, reconstruction_error=self.model.reconstruction_error)
            
            neg_ELBO_wt, latent_output_wt, _ = self.model.compute_loss(x_wt, stochastic_latent=False, kl_global_params_scale=0, beta=1)
            ELBO_WT, y_hat_dms_wt = -neg_ELBO_wt, latent_output_wt["y_hat_dms"]
            
            #ypred = (1 - self.training_parameters["model_mixing"])*(elbo - ELBO_WT) + (self.training_parameters["model_mixing"])*D
            ypred = elbo - ELBO_WT
            
            # Define Elbo based on dms target
            elbo_dmstarget = F.mse_loss(y_hat_dms, y_true_dms) + latent_output["KLD"]
            elbo_wt_dmstarget = y_hat_dms_wt.pow(2).mean() + latent_output_wt["KLD"]

            ypred_dmstarget = elbo_dmstarget - elbo_wt_dmstarget

            if not (dataloader_name in self.val_scores.keys()):
                self.val_scores[dataloader_name] = []
            if not (dataloader_name in self.val_scores_dms.keys()):
                self.val_scores_dms[dataloader_name] = []
            
            # Save batch scores for later
            self.val_scores[dataloader_name].append(torch.cat([ypred[:, None], batch["target"][:, None], batch["target_bin"][:, None], elbo[:, None], ELBO_WT[:, None], y_hat_dms[:,[0]], y_true_dms[:,[0]], ypred_dmstarget[:,None]], dim=1))

    def on_validation_epoch_end(self):
        if all([len(v)>0 for v in self.val_scores.values()]):
            # Aggregate the scores obtained on different datasets
            for idataset, (dataloader_name, protein_val_scores) in enumerate(self.val_scores.items()):
                # Concatenate and plot the DMS scores
                # Concatenate all the batch results 
                all_outputs = torch.cat(protein_val_scores).detach().cpu()
                
                predictions_elbo, targets_dms, targets_bin, elbo, ELBO_WT, predictions_latentoutput, targets_latentoutput, predictions_elbodmstarget = all_outputs.T
                
                thescores = get_dms_scores(predictions_elbo, targets_dms, targets_bin)
                thescores_latentoutput = get_dms_scores(predictions_latentoutput, targets_latentoutput, targets_bin)
                thescores_predictions_elbodmstarget = get_dms_scores(predictions_elbodmstarget, targets_latentoutput, targets_bin)
                
                log_dict = {**{"{}/{}".format(score_name, dataloader_name): score_value for score_name, score_value in thescores.items()},
                            **{"{}/{}/latent".format(score_name, dataloader_name): score_value for score_name, score_value in thescores_latentoutput.items()},
                            **{"{}/{}/dmsELBO".format(score_name, dataloader_name): score_value for score_name,score_value in thescores_predictions_elbodmstarget.items()}
                            }
                
                log_dict["hp/earlystop_metric"] = thescores["spearmanr"]

                if self.plot_elbos:
                    axes = self.val_score_figure[1]
                    axes[0].cla()
                    axes[0].scatter(targets_dms, predictions_elbo,label="ELBO")
                    axes[0].set_xlabel("DMS Targets")
                    axes[0].set_ylabel("Predictions")
                    axes[0].legend()
                    
                    axes[1].cla()
                    axes[1].scatter(targets_latentoutput, predictions_latentoutput, label="MLPlatent", c="darkred", s=10)
                    axes[1].set_xlabel("DMS Targets")
                    axes[1].set_ylabel("MLPlatent Predictions")

                    self.logger.experiment.add_figure("dms_y-yhat_plot/{}".format(dataloader_name), 
                        self.val_score_figure[0], self.the_training_step
                    )

                self.log_dict(log_dict)
            self.val_scores = {k: [] for k in self.trainer.val_dataloaders_info["name"] if k.startswith("dms")}
            self.val_scores_dms = {k: [] for k in self.trainer.val_dataloaders_info["name"] if k.startswith("dms")}

        # Move all the scores to cpu
        self.val_scores_msa = {k1:{k2:v.item() if not (isinstance(v,float)) else v for k2,v in self.val_scores_msa[k1].items()} for k1 in self.val_scores_msa.keys()}
        log_dict = pd.DataFrame(self.val_scores_msa).mean(1).to_dict()
        log_dict["tau"] = self.model.matVAE.tau
        self.log_dict(log_dict)
        ###self.val_scores_msa = {k:[] for k in self.trainer.val_dataloaders_info["name"] if not k.startswith("dms")}
    
    def on_before_optimizer_step(self, optimizer):
        """Saving gradient norms"""

        if (self.the_training_step>10) & (self.training_parameters['gradient_clip_norm']>0):
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.training_parameters['gradient_clip_norm'])
        
        if self.training_parameters['gradient_noise_std']>0:
            for param in self.parameters():
                if not (param.grad is None):# param.requires_grad:
                    noise_std = self.training_parameters['gradient_noise_std']#param.grad.std() * 10**(-self.training_parameters['gradient_noise_db']/20)
                    param.grad += torch.rand_like(param.grad) * noise_std

        if (self.training_parameters["log_grad_every"]>0) and ((self.the_training_step%self.training_parameters["log_grad_every"])==0):
            grad_log = {}

            if hasattr(self.model,"encoding_transformer"):
                if not (self.model.encoding_transformer.attn_layers[0].conv_Q.weight.grad is None):
                    grad_log["grad_norm2/transformer-first"] = self.model.encoding_transformer.attn_layers[0].conv_Q.weight.grad.norm(2)
                if not (self.model.encoding_transformer.fullyconnected[-1].linear_layers[-1].weight.grad is None):
                    grad_log["grad_norm2/transformer-last"] = self.model.encoding_transformer.fullyconnected[-1].linear_layers[-1].weight.grad.norm(2)

            if hasattr(self.model,"MLP_down"):
                if not (self.model.MLP_down.layers[0].weight.grad is None):
                    grad_log["grad_norm2/MLP_down-first"] = self.model.MLP_down.layers[0].weight.grad.norm(2)
                    grad_log["grad_norm2/MLP_down-last"] = self.model.MLP_down.layers[-1].weight.grad.norm(2)

            if hasattr(self.model,"vae_encoder"):  
                if not (self.model.vae_encoder.model.linear_layers[0].weight.grad is None):
                    grad_log["grad_norm2/vae_encoder-first"] = self.model.vae_encoder.model.linear_layers[0].weight.grad.norm(2)
                    grad_log["grad_norm2/vae_encoder-last"] = self.model.vae_encoder.model.linear_layers[-1].weight.grad.norm(2)
                
                if not (self.model.fc_mu_z.linear_layers[0].weight.grad is None):
                    grad_log["grad_norm2/vae_encoder_mu-first"] = self.model.fc_mu_z.linear_layers[0].weight.grad.norm(2)
                    grad_log["grad_norm2/vae_encoder_mu-last"] = self.model.fc_mu_z.linear_layers[-1].weight.grad.norm(2)

            if hasattr(self.model,"latent_to_dms"):
                if not (self.model.latent_to_dms.linear_layers[0].weight.grad is None):
                    grad_log["grad_norm2/latent_to_dms-first"]=self.model.latent_to_dms.linear_layers[0].weight.grad.norm(2)
                    grad_log["grad_norm2/latent_to_dms-last"]=self.model.latent_to_dms.linear_layers[-1].weight.grad.norm(2)
            self.log_dict(grad_log, on_step=True)
            grad_log={}
            if any([torch.isnan(p).any() for p in self.parameters()]):
                grad_log["hp/max_grad"] = torch.nan
                print("Found nan. Exit.")
                sys.exit(0)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], 
        lr=self.training_parameters['learning_rate'], 
        weight_decay = self.training_parameters['l2_regularization'],
        betas = self.training_parameters['betas'])

        if self.training_parameters['use_lr_scheduler']:
            scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=self.training_parameters['lr_scheduler_step_size'], gamma=self.training_parameters['lr_scheduler_gamma'])

            return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return optim


def annealing_factor(annealing_warm_up, training_step):
    """
    Annealing schedule of KL to focus on reconstruction error in early stages of training
    """
    if training_step < annealing_warm_up:
        return training_step / annealing_warm_up
    else:
        return 1

def get_dms_scores(predictions, targets, targets_binary):
    out = dict(r2=r2_score(predictions,targets), 
                spearmanr=_spearman_corrcoef_compute(predictions, targets),
                binary_auroc=binary_auroc(predictions,targets_binary.int()),
                binary_auprc=binary_auprc(predictions,targets_binary.int()),
                binary_precision=binary_precision(predictions,targets_binary.int()),
                binary_recall=binary_recall(predictions,targets_binary.int()),
                binary_f1_score=binary_f1_score(predictions,targets_binary),
                binary_accuracy=binary_accuracy(predictions,targets_binary))
    return out