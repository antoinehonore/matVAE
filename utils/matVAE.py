
import torch
import os 

from utils.mlp import MLP, init_all 
from utils.attention import MultiSelfAttention
from utils.encoders import simple_mlp_encoder
from utils.decoders import VAE_Bayesian_MLP_decoder, simple_mlp_decoder
from utils.activations import get_activation
from utils.priors import MogPrior, VampPrior, rho_to_logvar # , kl_divergence_two_gaussians,kl_divergence_gaussian_vs_mog
from utils.data import get_aa_dict, read_pdb

class VarLenMLPup(torch.nn.Module):
    def __init__(self, max_size, insize, layers_sizes, activation, dropout_p=0, layernorm=False):
        super(VarLenMLPup, self).__init__()
        self.layernorm = layernorm
        self.layers_sizes = layers_sizes + [1]
        self.insize = insize
        self.max_size = max_size
        
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
            
            y = torch.nn.functional.linear(   
                y.transpose(1, 2), weight=A_l_i, bias=bias_l_i
            ).transpose(1, 2)
            
            if i < len(self.layers) - 1:
                y = self.activation(y)
                y = self.dropout(y)
                if self.layernorm and (y.ndim > 1):
                    y = torch.nn.functional.layer_norm(y, normalized_shape=[y.shape[-1]])
        return y

class VarLenMLPdown(torch.nn.Module):
    def __init__(self, max_size, outsize, layers_sizes, activation, dropout_p=0, layernorm=False):
        super(VarLenMLPdown, self).__init__()
        self.layernorm=layernorm
        self.layers_sizes = [1] + layers_sizes
        self.outsize = outsize
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
                
                y = torch.nn.functional.linear(   
                    y.transpose(1, 2), weight=A_l_i, bias=bias_l_i
                ).transpose(1, 2)

                if i < len(self.layers) - 1:
                    y = self.activation(y)
                    y = self.dropout(y)
                    if self.layernorm and (y.ndim > 1):
                        y = torch.nn.functional.layer_norm(y, normalized_shape=[y.shape[-1]])
            
        return y

class matEncoderDecoder(torch.nn.Module):
    def __init__(self, hparams, decoder=True):
        super(matEncoderDecoder, self).__init__()
        self.hparams = hparams
        self.model_type = hparams["model_type"]
        if "skiptemperature" in hparams.keys():
            self.skiptemperature = hparams["skiptemperature"]
        else:
            self.skiptemperature=1

        if "transformers" in self.model_type:
            self.init_transformers(hparams,decoder=decoder)
        
        # LMAX network
        if "Lmax" in self.model_type:
            self.init_variable_length(hparams,decoder=decoder)

    def init_transformers(self, hparams, decoder=True):

        self.init_adjacency(hparams["protein_name"], hparams["transformer_use_structure"])
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

        init_all(self.encoding_transformer.fullyconnected, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["transformer_activation"]))
        if decoder:
            self.decoding_transformer = MultiSelfAttention(
                in_dim, key_dim, out_dim, num_heads, 
                num_layers=num_layers, kernel_size=kernel_size, attn_mask=self.adjacency_matrix if use_structure else None, 
                dropout_p=hparams["transformer_dropout_p"],activation=hparams["transformer_activation"],attnffn_num_layers=hparams["transformer_attnffn_num_layers"]
            )

            init_all(self.decoding_transformer.fullyconnected, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["transformer_activation"]))


    def init_variable_length(self, hparams, decoder=True):
        self.Lmax = hparams["variable_length_Lmax"]
        self.H = hparams["variable_length_H"]

        n_layers = hparams["variable_length_n_layers"]
        
        variable_length_layers = [i / (n_layers) for i in range(1, n_layers)][::-1]

        self.MLP_down = VarLenMLPdown(self.Lmax, self.H, variable_length_layers,   
                        get_activation(hparams["variable_length_activation"]), 
                        dropout_p=hparams["variable_length_dropout_p"],layernorm=hparams["layernorm"]
                        )
        init_all(self.MLP_down, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["variable_length_activation"]))
        if decoder:
            self.MLP_up = VarLenMLPup(self.Lmax, self.H, variable_length_layers[::-1], 
                            get_activation(hparams["variable_length_activation"]), 
                            dropout_p=hparams["variable_length_dropout_p"],layernorm=hparams["layernorm"]
                            )
            
            init_all(self.MLP_up, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["variable_length_activation"]))

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
        
    def encode(self,x):
        latent_output = {}
        
        L = x.shape[1]

        # Encoder
        if "transformers" in self.model_type:
            xout = self.encoding_transformer(x)
            x = xout
            latent_output["transformers_yout"] = xout#.detach()
        
        if "Lmax" in self.model_type:
            x = self.MLP_down(x)
            if hasattr(self,"MLP_up"):
                self.MLP_up.current_L = L
            latent_output["variable_length"] = x#.detach()

        if "mean" in self.model_type:
            self.xin = x
            x = torch.mean(x,dim=1,keepdim=True)

        return x, latent_output

    def decode(self,logits):
        if "mean" in self.model_type:
            logits = self.xin + logits
        
        if "Lmax" in self.model_type:
            logits = self.MLP_up(logits)
            self.MLP_up.current_L = None

        if "transformers" in self.model_type:
            logits = self.decoding_transformer(logits)
        return logits

class matVAE(torch.nn.Module):
    def __init__(self, hparams):
        super(matVAE, self).__init__()
        self.hparams = hparams

        # INIT encoder decoder 
        self.mat_encoder_decoder_mu = matEncoderDecoder(hparams)
        self.prior_type = hparams["vae_prior_type"]
        self.model_type = hparams["model_type"]

        self.vae_input_size = hparams["vae_input_size"]
        self.include_temperature_scaler = hparams["include_temperature_scaler"]

        if self.include_temperature_scaler > 0:
            self.temperature_scaler_mean = torch.nn.Parameter(self.include_temperature_scaler*torch.ones(1))

        self.prior_temperature_scaler = hparams["vae_prior_temperature_scaler"]
        self.prior_init_scaler = hparams["vae_prior_init_scaler"]
        self.enc_layers = hparams["vae_enc_layers"]
        self.dec_layers = self.enc_layers[::-1]
        self.z_dim = hparams["vae_z_dim"] if not ("entropy" in self.prior_type) else hparams["vae_num_prior_components"]

        self.stochastic_latent = hparams["vae_stochastic_latent"]
        self.vae_dropout_p = hparams["vae_dropout_p"]
        self.vae_decoder_include_sparsity = hparams["vae_decoder_include_sparsity"]
        self.vae_decoder_convolve_output = hparams["vae_decoder_convolve_output"]
        self.vae_decoder_bayesian = hparams["vae_decoder_bayesian"]
        
        vae_activation = get_activation(hparams["vae_activation"])

        self.vae_encoder_mu = simple_mlp_encoder(self.vae_input_size, self.z_dim, self.enc_layers, vae_activation, dropout_p=self.vae_dropout_p,
            layernorm=hparams["layernorm"],skipconnections= hparams["skipconnections"], skiptemperature=hparams["skiptemperature"])
        #self.vae_encoder_rho = simple_mlp_encoder(self.vae_input_size, self.z_dim, self.enc_layers, vae_activation, dropout_p=self.vae_dropout_p,layernorm=hparams["layernorm"],skipconnections= hparams["skipconnections"])

        init_all(self.vae_encoder_mu, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["vae_activation"]))
        #init_all(self.vae_encoder_rho, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["vae_activation"]))
        #self.layer_norm = torch.nn.LayerNorm()
        self.mu_bias_init = 0.1
        self.log_var_bias_init =  -10.0
        enc_out_dim = self.vae_encoder_mu.model.linear_layers[-1].out_features

        if ("mog" in self.prior_type) or ("vamp" in self.prior_type):
            self.fc_mu_z = MLP(enc_out_dim, [self.z_dim]*hparams["vae_latent_n_layers"], self.z_dim, get_activation(hparams["vae_latent_activation"]), 
                                layernorm=hparams["layernorm"],skipconnections= hparams["skipconnections"], skiptemperature=hparams["skiptemperature"])
            
            self.fc_rho_z = MLP(enc_out_dim, [self.z_dim]*hparams["vae_latent_n_layers"], self.z_dim, get_activation(hparams["vae_latent_activation"]), 
                                layernorm=hparams["layernorm"],skipconnections= hparams["skipconnections"], skiptemperature=hparams["skiptemperature"])
            
            self.fc_w_mixture = MLP(enc_out_dim, [self.z_dim]*hparams["vae_latent_n_layers"],  hparams["vae_num_prior_components"], get_activation(hparams["vae_latent_activation"]), 
                        layernorm=hparams["layernorm"], skipconnections= hparams["skipconnections"], skiptemperature=hparams["skiptemperature"])

            init_all(self.fc_mu_z, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["vae_latent_activation"]))
            init_all(self.fc_rho_z, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["vae_latent_activation"]))
        
        # DECODER
        params = dict()
        params['seq_len'] = hparams["L"] if not ("Lmax" in hparams["model_type"]) else hparams["variable_length_H"]
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
            init_all(self.vae_decoder, torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(hparams["vae_activation"]))

        self.prototype_shape = (hparams["L"], hparams["d"])

        self.num_prior_components = hparams["vae_num_prior_components"]

        if ("mog" in self.prior_type):
            self.Prior = MogPrior(self.z_dim, self.num_prior_components, self.prior_type, init_scaler=self.prior_init_scaler)

        elif "vamp" in self.prior_type:
            self.Prior = VampPrior(self.num_prior_components, self.prior_type, prototype_shape=self.prototype_shape ,temperature_scaler=self.prior_init_scaler)
        
        if "entropy" in self.prior_type:
            self.tau = torch.nn.Parameter(torch.tensor([1.]), requires_grad=hparams["latent_train_tau"])

    def encode(self, x, stochastic_latent=True):
        # Latent space
        z_encoding, latent_output = self.mat_encoder_decoder_mu.encode(x)
        self.start_dim, self.end_dim = z_encoding.shape[1:] # Store L,d for decoding
        z_encoding = torch.flatten(z_encoding, start_dim=1, end_dim=2)

        z_pre_latent = self.vae_encoder_mu(z_encoding)
        
        if ("mog" in self.prior_type) or ("vamp" in self.prior_type):
            logits_w_mixture = self.fc_w_mixture(z_pre_latent)
            y_mixture = logits_w_mixture.softmax(-1)

            mu  =  self.fc_mu_z(torch.cat([z_pre_latent, y_mixture], dim=-1))
            log_var = rho_to_logvar(self.fc_rho_z(z_pre_latent))

            if self.stochastic_latent and stochastic_latent:
                z = self.sample_latent(mu, log_var)
            else:
                z = mu
            latent_output = {**latent_output, "mu":mu, "log_var":log_var, "latent_to_dms_input":mu}
        
        elif ("entropy" in self.prior_type):
            z0 = torch.nn.functional.layer_norm(z_pre_latent, normalized_shape=z_pre_latent.shape[1:])#self.fc_w_mixture(z_pre_latent)
            tau = torch.nn.functional.softplus(self.tau)

            z_soft = torch.nn.functional.softmax(z0 / tau,dim=-1)
            
            latent_output= {**latent_output,"mu":z_soft,"log_var":z_soft}
            
            if self.stochastic_latent and stochastic_latent:
                z = torch.nn.functional.gumbel_softmax(z0, tau=tau, dim=-1)
            else:
                z = z_soft
            
            latent_output["latent_to_dms_input"] = z_soft
        latent_output = {**latent_output, "z": z}

        #  KL div
        if ("mog" in self.prior_type):
            kl = self.Prior.kl_div(mu, log_var)
        
        elif ("entropy" in self.prior_type):
            # to minimize
            kl = (-(z_soft*z_soft.log()).sum(1)) # Not KL but entropy
            #kl = self.Prior.kl_div(mu, log_var) #+ entropy_
            
        elif "vamp" in self.prior_type:
            prototypes_in = self.Prior.prepare_prototypes()
            prot_enc = self.mat_encoder_decoder_mu.encode(prototypes_in)

            prototypes_mu, _  =  self.fc_mu_z(prot_enc)
            prototypes_log_var = rho_to_logvar(self.fc_rho_z(prot_enc))
            
            kl = self.Prior.kl_div(mu, log_var, prototypes_mu, prototypes_log_var)
        
        latent_output["KLD"] = kl
        
        if self.vae_decoder.bayesian_decoder:
            latent_output["KLD_params"] = self.vae_decoder.get_KL_div()

        return z, latent_output

    def decode(self, z):

        # Decoder
        logits = self.vae_decoder(z)
        
        logits = torch.unflatten(logits, -1, (self.start_dim, self.end_dim))
        self.start_dim, self.end_dim = None,None

        logits = self.mat_encoder_decoder_mu.decode(logits)

        if self.include_temperature_scaler > 0:
            logits = torch.log(1.0+torch.exp( self.temperature_scaler_mean)) * logits
        return logits

    def sample_latent(self, mu, log_var):
        """
            Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu, device=mu.device)
        z = torch.exp(0.5*log_var) * eps + mu
        return z
