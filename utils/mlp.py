import torch


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


class MLP(torch.nn.Module):
    def __init__(self,input_size, layers_sizes, output_size, activation, dropout_p=0, layernorm=False, skipconnections=False, skiptemperature=False):
        super(MLP,self).__init__()
        self.layernorm = layernorm
        self.input_size, self.layers_sizes, self.output_size, self.activation = input_size, layers_sizes, output_size, activation
        #self.dropout_p = dropout_p
        self.dropout_function = torch.nn.Dropout(dropout_p)
        self.activation_function = activation()
        self.skipconnections = skipconnections
        self.skiptemperature = skiptemperature

        self.linear_layers = []
        if self.skipconnections:
            self.linear_layers_skip = []
        
        for i in range(len(layers_sizes)+1):
            layer_in_size = input_size if i==0 else layers_sizes[i-1]
            layer_out_size = layers_sizes[i] if i < (len(layers_sizes)) else output_size

            self.linear_layers.append(torch.nn.Linear(layer_in_size, layer_out_size))
            if self.skipconnections:
                if layer_in_size == layer_out_size:
                    self.linear_layers_skip.append(torch.nn.Linear(layer_in_size, layer_out_size))
                else:
                    self.linear_layers_skip.append(None)
                
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        if self.skiptemperature:
            self.skiptemperature_params = torch.nn.ParameterList([torch.zeros(1) for _ in range(len(self.linear_layers))])
            self.skiptemperature_params.requires_grad_(False)
            
        
        if self.skipconnections:
            self.linear_layers_skip = torch.nn.ModuleList(self.linear_layers_skip)
    
    #@torch.compile(mode="default")
    def forward(self, x):
        """An MLP with skipconnections, layer normalizations, and dropout
            
            x = dropout( x )            // Not at the first layer  
            x = linear( x )
            x = normalize( x )          // If required and (not at the last layer unless there is a skip connection)
            if skipconnections
            |    y = activation( x )
            |    y = linear( y )
            |    y = normalize ( y )    // Always
            |    x = x + y
            x = activation( x )         // Not at the last layer
            """
        for i in range(len(self.linear_layers)):
            is_last_layer = i == (len(self.linear_layers)-1)

            if i > 0:
                x = self.dropout_function(x)

            # Linear
            xout = self.linear_layers[i](x)
            
            # Normalize
            possible_to_normalize = (xout.ndim == 2) and (xout.shape[1] != 1)
            possible_to_skip = self.skipconnections and (xout.shape == x.shape)

            if self.layernorm and possible_to_normalize:
                # If we are not at the last layer, or we required skip connections 
                if (not is_last_layer) or possible_to_skip:# (self.skipconnections):
                    xout = torch.nn.functional.layer_norm(xout, normalized_shape=xout.shape[1:])
            
            if possible_to_skip:
                # Activation
                xout = self.activation_function(xout)
                
                # Linear
                xout = self.linear_layers_skip[i](xout)

                # Normalize if not last layer
                if (not is_last_layer):
                    xout = torch.nn.functional.layer_norm(xout, normalized_shape=xout.shape[1:])

                # Residual
                tau = torch.nn.functional.sigmoid(self.skiptemperature_params[i])
                x = xout*(1-tau) + x*tau
            else:
                x = xout

            if not is_last_layer:
                # Activation
                x = self.activation_function(x)
        return x
