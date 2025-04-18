import torch
import math
from packaging import version
from functools import partial

def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = torch.nn.functional.gelu

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

def _silu_python(x):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """
    return x * torch.sigmoid(x)

if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = torch.nn.functional.silu


def _mish_python(x):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    return x * torch.tanh(torch.nn.functional.softplus(x))

from packaging import version
if version.parse(torch.__version__) < version.parse("1.9"):
    mish = _mish_python
else:
    mish = torch.nn.functional.mish


def linear_act(x):
    return x

def squared_relu(x):
    """
    Squared ReLU variant that is fastest with Pytorch.
    """
    x = torch.nn.functional.relu(x)
    return x*x

def squared_relu_xla(x):
    """
    Squared ReLU variant that is fastest with JAX.
    """
    x = torch.nn.functional.relu(x)
    return x**2


class ActivationWrapper(torch.nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.activation = activation
        self.activation_fun = tranception_ACT2FN[activation]

        pass
    def forward(self,x):
        return self.activation_fun(x)
    def extra_repr(self) -> str:
            return self.activation

tranception_ACT2FN = {
    "relu": torch.nn.functional.relu,
    "silu": silu,
    "selu": torch.nn.SELU(),
    "leakyrelu": torch.nn.LeakyReLU(),
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
    "squared_relu": squared_relu,
    "squared_relu_xla": squared_relu_xla,
}


def get_activation(activation_string):
    if activation_string in tranception_ACT2FN:
        func = partial(ActivationWrapper,activation_string)
        return func
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(tranception_ACT2FN.keys())}")