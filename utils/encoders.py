import torch

from utils.mlp import MLP

class simple_mlp_encoder(torch.nn.Module):
    def __init__(self, input_size, z_dim, layers_sizes, activation, dropout_p = 0, layernorm=False, skipconnections=False):
        super(simple_mlp_encoder, self).__init__()
        self.input_size=input_size
        self.z_dim = z_dim
        self.layers_sizes = layers_sizes
        self.activation = activation  ## tranception_ACT2FN[activation]
        self.dropout_p = dropout_p
        
        self.model = MLP(self.input_size,self.layers_sizes,self.z_dim,self.activation,dropout_p=self.dropout_p,layernorm=layernorm,skipconnections=skipconnections)
    def forward(self, batch):
        return self.model(batch)
    