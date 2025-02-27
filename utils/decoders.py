import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import tranception_ACT2FN
from utils.mlp import MLP
from torch import Tensor
import numpy as np
from scipy.special import erfinv
#from utils import get_path
import logging

class VAE_Bayesian_MLP_decoder(torch.nn.Module):
    """
    Bayesian MLP decoder class for the VAE model.
    """
    def __init__(self, params):
        """
        Required input parameters:
        - seq_len: (Int) Sequence length of sequence alignment
        - alphabet_size: (Int) Alphabet size of sequence alignment (will be driven by the data helper object)
        - hidden_layers_sizes: (List) List of the sizes of the hidden layers (all DNNs)
        - z_dim: (Int) Dimension of latent space
        - first_hidden_nonlinearity: (Str) Type of non-linear activation applied on the first (set of) hidden layer(s)
        - last_hidden_nonlinearity: (Str) Type of non-linear activation applied on the very last hidden layer (pre-sparsity)
        - dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
        - convolve_output: (Bool) Whether to perform 1d convolution on output (kernel size 1, stide 1)
        - convolution_depth: (Int) Size of the 1D-convolution on output
        - include_temperature_scaler: (Bool) Whether we apply the global temperature scaler
        - include_sparsity: (Bool) Whether we use the sparsity inducing scheme on the output from the last hidden layer
        - num_tiles_sparsity: (Int) Number of tiles to use in the sparsity inducing scheme (the more the tiles, the stronger the sparsity)
        - bayesian_decoder: (Bool) Whether the decoder is bayesian or not
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = params['seq_len']
        self.alphabet_size = params['alphabet_size']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim = params['z_dim']
        self.bayesian_decoder = params["vae_decoder_bayesian"]
        self.dropout_proba = params['dropout_proba']
        self.convolve_output = params['convolve_output']
        self.convolution_depth = params['convolution_output_depth']
        self.include_temperature_scaler = params['include_temperature_scaler']
        self.include_sparsity = params['include_sparsity']
        self.num_tiles_sparsity = params['num_tiles_sparsity']
                
        self.mu_bias_init = 0.1
        self.logvar_init = -10.0
        self.logit_scale_p = 0.001
        
        self.hidden_layers_mean=nn.ModuleDict()
        if self.bayesian_decoder:
            self.hidden_layers_log_var=nn.ModuleDict()

        for layer_index in range(len(self.hidden_layers_sizes)):
            if layer_index==0:
                self.hidden_layers_mean[str(layer_index)] = nn.Linear(self.z_dim, self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers_mean[str(layer_index)].bias, self.mu_bias_init)
                if self.bayesian_decoder:
                    self.hidden_layers_log_var[str(layer_index)] = nn.Linear(self.z_dim, self.hidden_layers_sizes[layer_index])
                    nn.init.constant_(self.hidden_layers_log_var[str(layer_index)].weight, self.logvar_init)
                    nn.init.constant_(self.hidden_layers_log_var[str(layer_index)].bias, self.logvar_init)
            else:
                self.hidden_layers_mean[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index-1],self.hidden_layers_sizes[layer_index])
                nn.init.constant_(self.hidden_layers_mean[str(layer_index)].bias, self.mu_bias_init)
                if self.bayesian_decoder:
                    self.hidden_layers_log_var[str(layer_index)] = nn.Linear(self.hidden_layers_sizes[layer_index-1],self.hidden_layers_sizes[layer_index])
                    nn.init.constant_(self.hidden_layers_log_var[str(layer_index)].weight, self.logvar_init)
                    nn.init.constant_(self.hidden_layers_log_var[str(layer_index)].bias, self.logvar_init)

        if params['first_hidden_nonlinearity'] == 'relu':
            self.first_hidden_nonlinearity = nn.ReLU()
        elif params['first_hidden_nonlinearity'] == 'tanh':
            self.first_hidden_nonlinearity = nn.Tanh()
        elif params['first_hidden_nonlinearity'] == 'selu':
            self.first_hidden_nonlinearity = nn.SELU()
        elif params['first_hidden_nonlinearity'] == 'sigmoid':
            self.first_hidden_nonlinearity = nn.Sigmoid()
        elif params['first_hidden_nonlinearity'] == 'elu':
            self.first_hidden_nonlinearity = nn.ELU()
        elif params['first_hidden_nonlinearity'] == 'linear':
            self.first_hidden_nonlinearity = nn.Identity()
        
        if params['last_hidden_nonlinearity'] == 'relu':
            self.last_hidden_nonlinearity = nn.ReLU()
        elif params['last_hidden_nonlinearity'] == 'selu':
            self.last_hidden_nonlinearity = nn.SELU()
        elif params['last_hidden_nonlinearity'] == 'tanh':
            self.last_hidden_nonlinearity = nn.Tanh()
        elif params['last_hidden_nonlinearity'] == 'sigmoid':
            self.last_hidden_nonlinearity = nn.Sigmoid()
        elif params['last_hidden_nonlinearity'] == 'elu':
            self.last_hidden_nonlinearity = nn.ELU()
        elif params['last_hidden_nonlinearity'] == 'linear':
            self.last_hidden_nonlinearity = nn.Identity()

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

        if self.convolve_output:
            self.output_convolution_mean = nn.Conv1d(in_channels=self.convolution_depth,out_channels=self.alphabet_size,kernel_size=1,stride=1,bias=False)
            self.output_convolution_log_var = nn.Conv1d(in_channels=self.convolution_depth,out_channels=self.alphabet_size,kernel_size=1,stride=1,bias=False)
            nn.init.constant_(self.output_convolution_log_var.weight, self.logvar_init)
            self.channel_size = self.convolution_depth
        else:
            self.channel_size = self.alphabet_size
        
        if self.include_sparsity:
            self.sparsity_weight_mean = nn.Parameter(torch.zeros(int(self.hidden_layers_sizes[-1]/self.num_tiles_sparsity), self.seq_len))
            if self.bayesian_decoder:
                self.sparsity_weight_log_var = nn.Parameter(torch.ones(int(self.hidden_layers_sizes[-1]/self.num_tiles_sparsity), self.seq_len))
                nn.init.constant_(self.sparsity_weight_log_var, self.logvar_init)

        self.last_hidden_layer_weight_mean = nn.Parameter(torch.zeros(self.channel_size * self.seq_len,self.hidden_layers_sizes[-1]))
        nn.init.xavier_normal_(self.last_hidden_layer_weight_mean) #Glorot initialization
        
        if self.bayesian_decoder:
            self.last_hidden_layer_weight_log_var = nn.Parameter(torch.zeros(self.channel_size * self.seq_len,self.hidden_layers_sizes[-1]))
            nn.init.constant_(self.last_hidden_layer_weight_log_var, self.logvar_init)
        
        self.last_hidden_layer_bias_mean = nn.Parameter(torch.zeros(self.alphabet_size * self.seq_len))
        nn.init.constant_(self.last_hidden_layer_bias_mean, self.mu_bias_init)
        
        
        if self.bayesian_decoder:
            self.last_hidden_layer_bias_log_var = nn.Parameter(torch.zeros(self.alphabet_size * self.seq_len))
            nn.init.constant_(self.last_hidden_layer_bias_log_var, self.logvar_init)
        
        if self.include_temperature_scaler:
            self.temperature_scaler_mean = nn.Parameter(torch.ones(1))
            if self.bayesian_decoder:
                self.temperature_scaler_log_var = nn.Parameter(torch.ones(1) * self.logvar_init) 
        if self.bayesian_decoder:
            self.get_KL_div()
            
    def sampler(self, mean, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mean).to(self.device)
        z = torch.exp(0.5*log_var) * eps + mean
        return z

    def forward(self, z):
        batch_size = z.shape[0]
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(z)
        else:
            x = z

        for layer_index in range(len(self.hidden_layers_sizes)-1):
            if self.bayesian_decoder:
                layer_i_weight = self.sampler(self.hidden_layers_mean[str(layer_index)].weight, self.hidden_layers_log_var[str(layer_index)].weight)
                layer_i_bias = self.sampler(self.hidden_layers_mean[str(layer_index)].bias, self.hidden_layers_log_var[str(layer_index)].bias)
            else:
                layer_i_weight = self.hidden_layers_mean[str(layer_index)].weight
                layer_i_bias = self.hidden_layers_mean[str(layer_index)].bias
            x = self.first_hidden_nonlinearity(F.linear(x, weight=layer_i_weight, bias=layer_i_bias))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        last_index = len(self.hidden_layers_sizes)-1
        if self.bayesian_decoder:
            last_layer_weight = self.sampler(self.hidden_layers_mean[str(last_index)].weight, self.hidden_layers_log_var[str(last_index)].weight)
            last_layer_bias = self.sampler(self.hidden_layers_mean[str(last_index)].bias, self.hidden_layers_log_var[str(last_index)].bias)
        else:
            last_layer_weight = self.hidden_layers_mean[str(last_index)].weight
            last_layer_bias = self.hidden_layers_mean[str(last_index)].bias
        
        x = self.last_hidden_nonlinearity(F.linear(x, weight=last_layer_weight, bias=last_layer_bias))
        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)
        
        if self.bayesian_decoder:
            W_out = self.sampler(self.last_hidden_layer_weight_mean, self.last_hidden_layer_weight_log_var)
            b_out = self.sampler(self.last_hidden_layer_bias_mean, self.last_hidden_layer_bias_log_var)
        else:
            W_out = self.last_hidden_layer_weight_mean#, self.last_hidden_layer_weight_log_var)
            b_out = self.last_hidden_layer_bias_mean#, self.last_hidden_layer_bias_log_var)

        if self.convolve_output:
            if self.bayesian_decoder:
                output_convolution_weight = self.sampler(self.output_convolution_mean.weight, self.output_convolution_log_var.weight)
            else: 
                output_convolution_weight = self.output_convolution_mean.weight
            W_out = torch.mm(W_out.view(self.seq_len * self.hidden_layers_sizes[-1], self.channel_size), 
                                    output_convolution_weight.view(self.channel_size,self.alphabet_size)) #product of size (H * seq_len, alphabet)
            
        if self.include_sparsity:
            if self.bayesian_decoder:
                sparsity_weights = self.sampler(self.sparsity_weight_mean,self.sparsity_weight_log_var)
            else:
                sparsity_weights = self.sparsity_weight_mean
            
            sparsity_tiled = sparsity_weights.repeat(self.num_tiles_sparsity,1) 
            sparsity_tiled = nn.Sigmoid()(sparsity_tiled).unsqueeze(2) 

            W_out = W_out.view(self.hidden_layers_sizes[-1], self.seq_len, self.alphabet_size) * sparsity_tiled
        
        W_out = W_out.view(self.seq_len * self.alphabet_size, self.hidden_layers_sizes[-1])
        
        x = F.linear(x, weight=W_out, bias=b_out)

        if self.include_temperature_scaler:
            if self.bayesian_decoder:
                temperature_scaler = self.sampler(self.temperature_scaler_mean,self.temperature_scaler_log_var)
            else:
                temperature_scaler = self.temperature_scaler_mean
            x = torch.log(1.0+torch.exp(temperature_scaler)) * x
        
        return x

    @staticmethod
    def _KLD_diag_gaussians(mu, log_sigma, prior_mu=0., prior_log_sigma=0.) -> Tensor:
        """ KL divergence between two Diagonal Gaussians """
        return prior_log_sigma - log_sigma + \
               0.5 * (torch.exp(2.*log_sigma) + torch.square(mu - prior_mu)) * torch.exp(-2.*prior_log_sigma*torch.ones(1,device=mu.device)) - 0.5

    def _get_KLD_from_param(self, mu, log_sigma) -> Tensor:
        """ KL divergence between two Diagonal Gaussians """
        return torch.sum(self._KLD_diag_gaussians(mu.flatten(), log_sigma.flatten(), 0., 0.))

    def get_KL_div(self):
        KL_div = 0
        for k in self.hidden_layers_mean.keys():
            KL_div += self._get_KLD_from_param(self.hidden_layers_mean[k].weight, 0.5*self.hidden_layers_log_var[k].weight)
            KL_div += self._get_KLD_from_param(self.hidden_layers_mean[k].bias, 0.5*self.hidden_layers_log_var[k].bias)
        
        if self.convolve_output:
            KL_div += self._get_KLD_from_param(self.output_convolution_mean.weight, self.output_convolution_log_var.weight)
        
        KL_div += self._get_KLD_from_param(self.last_hidden_layer_weight_mean, 0.5*self.last_hidden_layer_weight_log_var)
        KL_div += self._get_KLD_from_param(self.last_hidden_layer_bias_mean, 0.5*self.last_hidden_layer_bias_log_var)
        
        #KL_div += self._get_KLD_from_param(self.W_out_mu, self.W_out_log_sigma)
        #KL_div += self._get_KLD_from_param(self.b_out_mu, self.b_out_log_sigma)
        #KL_div += self._get_KLD_from_param(self.final_pwm_scale_mu, self.final_pwm_scale_log_sigma)

        # Use a continuous relaxation of a spike and slab prior with a logit normit scale distribution
        #KL_div_scale = -self._KLD_diag_gaussians(self.W_out_scale_mu, self.W_out_scale_log_sigma,
        #                                         self.scale_prior_mu, self.scale_prior_log_sigma)
        #KL_div += torch.sum(KL_div_scale)

        return KL_div

class simple_mlp_decoder(torch.nn.Module):
    def __init__(self, input_size, z_dim, layers_sizes, activation, dropout_p=0):
        super(simple_mlp_decoder, self).__init__()
        self.input_size = input_size
        self.z_dim = z_dim
        self.layers_sizes = layers_sizes
        self.dropout_p = dropout_p
        self.activation = activation#tranception_ACT2FN[activation]
        

        # Sequence of dropout -> linear -> activation
        # Remove the first dropout, and last activation
        # Make sure that the first linear has input dimension> z_dim
        # Make sure that the last linear has input dimenesion> input_size
        self.model = MLP(self.z_dim, self.layers_sizes, self.input_size, self.activation,dropout_p=self.dropout_p)

        #self.model = torch.nn.Sequential(*sum([[torch.nn.Dropout(self.dropout_p),
        #                                            torch.nn.Linear(self.z_dim if i==0 else self.layers_sizes[i-1],
        #                                            self.layers_sizes[i] if i<len(self.layers_sizes) else self.input_size
        #                                            ),
        #                                        self.activation(),
        #                                       ]
        #                                   for i in range(len(self.layers_sizes)+1)], [])[:-1])

    def forward(self,batch):
        return self.model(batch)

