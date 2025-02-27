import torch
import torch.nn as nn
from utils.activations import get_activation
from utils.mlp import MLP

the_feature_map=lambda x: x

class SelfAttentionLayer(torch.nn.Module):

    def __init__(self, in_dim, key_dim, out_dim, num_heads, kernel_size=1, dropout_p=0, linear=False):
        super(SelfAttentionLayer, self).__init__()
        #assert (out_dim % num_heads) == 0,     "Make sure out_dim is divisible by num_heads"
        #assert (kernel_size==1) or ((kernel_size % 2) == 0), "Make sure kernel_size>=2 is even, for padding"
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        
        #assert (self.key_dim % self.num_heads)==0, "Num head should divide, key_dim"
        #assert (self.out_dim % self.num_heads)==0, "Num head should divide, out_dim"
        self.conv_Q = torch.nn.Conv1d(self.in_dim, self.key_dim*self.num_heads, kernel_size=self.kernel_size, bias=True, padding=self.kernel_size//2)
        self.conv_K = torch.nn.Conv1d(self.in_dim, self.key_dim*self.num_heads, kernel_size=self.kernel_size, bias=True, padding=self.kernel_size//2)
        self.conv_V = torch.nn.Conv1d(self.in_dim, self.out_dim*self.num_heads, kernel_size=self.kernel_size, bias=True, padding=self.kernel_size//2)
        self.feature_map = the_feature_map(self.key_dim)
        self.eps = 1e-6

        if linear:
            self.attn = self.linear_scaled_dot_product#
            #self.attn_mask = None
        else:
            self.attn = torch.nn.functional.scaled_dot_product_attention
            #self.attn_mask = attn_mask

        self.in_net = nn.Sequential(*[
            torch.nn.Conv1d(in_dim, key_dim, kernel_size=(self.kernel_size,), stride=(1,), padding=(self.kernel_size//2,)),
            torch.nn.ReLU(), 
            torch.nn.Conv1d(key_dim, key_dim, kernel_size=(self.kernel_size,), stride=(1,), padding=(self.kernel_size//2,)), 
            ])

        if self.num_heads > 1:
            self.fc_output = nn.Linear(self.out_dim*self.num_heads, self.out_dim)
        
        self.internal_metrics = {}
    """
    from fast_transformers.attention import LinearAttention
    from fast_transformers.feature_maps import elu_feature_map,ActivationFunctionFeatureMap

    selu_feature_map = ActivationFunctionFeatureMap.factory(
        lambda x: torch.nn.functional.selu(x)
    )
    the_feature_map = ActivationFunctionFeatureMap.factory(
        lambda x: torch.nn.functional.relu(x)
    )


    def linear_scaled_dot_product(self, queries, keys, values, attn_mask=None):
            Q = self.feature_map(queries).view(*queries.shape[:2],self.num_heads,self.key_dim)#[:,:,None,:]
            K = self.feature_map(keys).view(*keys.shape[:2],self.num_heads,self.key_dim) #[:,:,None,:]
            values = values.view(*values.shape[:2],self.num_heads,self.out_dim)

            # Compute the KV matrix, namely the dot product of keys and values so
            # that we never explicitly compute the attention matrix and thus
            # decrease the complexity
            KV = torch.einsum("nthd,nthm->nhmd", K, values)

            # Compute the normalizer
            Z = 1/(torch.einsum("nthd,nhd->nth", Q, K.sum(dim=1))+self.eps)

            # Finally compute and return the new values
            V = torch.einsum("nthd,nhmd,nth->nthm", Q, KV, Z)
            V = V.flatten(start_dim=2,end_dim=3)
            if self.num_heads>1:
                V = self.fc_output(V)
            return V
    """
    
    def forward(self, x, attn_mask=None, xref=None):
        """x are (N, L, D)"""
        if xref is None:
            y = x
        else:
            y = xref

        #x = self.in_net(x.transpose(1,2)).transpose(1,2)
        #y = self.in_net(y.transpose(1,2)).transpose(1,2)
        
        Q = self.conv_Q(x.transpose(1,2)).transpose(1,2)
        K = self.conv_K(y.transpose(1,2)).transpose(1,2)
        V = self.conv_V(y.transpose(1,2)).transpose(1,2)
        y = self.attn(Q, K, V, attn_mask=attn_mask if not (attn_mask is None) else attn_mask)
        return y

class MultiSelfAttention(torch.nn.Module):
    def __init__(self, in_dim, key_dim, out_dim, num_heads, num_layers, attnffn_num_layers, linear=False,
                    kernel_size=1, dropout_p=0, output_size=None, activation="relu", attn_mask=None):
        super(MultiSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.attnffn_num_layers = attnffn_num_layers
        
        self.only_attn = attnffn_num_layers == 0
        
        self.register_buffer("attn_mask",attn_mask)

        self.attn_layers = torch.nn.ModuleList(
            [SelfAttentionLayer(
                in_dim if i==0 else key_dim, key_dim, 
                out_dim if i == (num_layers-1) else key_dim, 
                num_heads, linear= False, 
                kernel_size=kernel_size, dropout_p=dropout_p
                ) for i in range(num_layers)]
        )

        self.activation = get_activation(activation)
        self.skiptemperature_params_first = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(len(self.attn_layers))])
        self.skiptemperature_params_second = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(len(self.attn_layers))])

        self.fullyconnected = torch.nn.ModuleList(
            
            MLP(key_dim, [key_dim]*self.attnffn_num_layers, out_dim, self.activation,dropout_p=dropout_p) for i in range(self.num_layers)
        )
        
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_size = output_size
        self.init()

    def init(self):

        # Initialize so that the convolutions return the input (when the inputs are one hot vectors)
        for i in range(len(self.attn_layers)):
            theshape = self.attn_layers[i].conv_Q.weight.shape
            if theshape[0] == theshape[1]:
                self.attn_layers[i].conv_Q.weight = nn.Parameter((torch.eye( theshape[0]).float().view(theshape)))
            theshape = self.attn_layers[i].conv_K.weight.shape
            if theshape[0] == theshape[1]:
                self.attn_layers[i].conv_K.weight = nn.Parameter((torch.eye( theshape[0]).float().view(theshape)))
            theshape = self.attn_layers[i].conv_V.weight.shape
            if theshape[0] == theshape[1]:
                self.attn_layers[i].conv_V.weight = nn.Parameter((torch.eye( theshape[0]).float().view(theshape)))

            torch.nn.init.constant_(self.attn_layers[i].conv_Q.bias, 0.)
            torch.nn.init.constant_(self.attn_layers[i].conv_K.bias, 0.)
            torch.nn.init.constant_(self.attn_layers[i].conv_V.bias, 0.)
    
    def forward(self, x, xref=None):
        for i in range(len(self.attn_layers)):
            attn_out = self.attn_layers[i](x, attn_mask=self.attn_mask, xref=None)
            
            if self.only_attn:
                x = attn_out
                if i < len(self.attn_layers):
                    x = self.activation()(x)
            else:
                # Skip connection + dropout
                tau = torch.nn.functional.sigmoid(self.skiptemperature_params_first[i])
                x = x*tau + self.dropout(attn_out)*(1-tau)

                # Normalize
                x = torch.nn.functional.layer_norm(x, [x.shape[-1]])
                
                # Skip connection + FFN layer
                tau = torch.nn.functional.sigmoid(self.skiptemperature_params_second[i])
                x = x*tau + self.fullyconnected[i](x)*(1-tau)

                # Normalized
                if i<(len(self.attn_layers)-1):
                    x = torch.nn.functional.layer_norm(x, [x.shape[-1]])

        return x