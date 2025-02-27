
import torch
import math

@torch.compile(mode="default")
def kl_divergence_two_gaussians(mu0, logvar0, mu1, logvar1):
    return 0.5 * torch.sum(logvar1 - logvar0 - 1 + torch.exp(logvar0 - logvar1) + (mu0 - mu1).pow(2) * torch.exp(-logvar1), dim=1)

@torch.compile(mode="default")
def rho_to_logvar(rho):
    return torch.log(torch.log(torch.exp(rho) + 1))

@torch.compile(mode="default")
def var_to_rho(var):
    return torch.log(torch.exp(var) - 1)

#@torch.compile(mode="default")
def kl_divergence_gaussian_vs_mog(mu, log_var, means, log_vars):
    """
    mu/log_var      of size (batch_size, dimension)
    means/log_vars  of size (1, dimension, num_components) 
    """
    normalizing_constant = math.log(means.shape[-1])
    kl_divs = kl_divergence_two_gaussians(
        mu.unsqueeze(-1), log_var.unsqueeze(-1), means, log_vars
        )
    kl = normalizing_constant - torch.logsumexp(- kl_divs, dim=1)
    return kl, kl_divs


class VampPrior(torch.nn.Module):
    def __init__(self, num_components, prior_type, prototype_shape=None, temperature_scaler=1):
        super(VampPrior, self).__init__()
        self.prior_type=prior_type
        self.temperature_scaler=temperature_scaler
        self.num_components = num_components
        self.prototype_shape = prototype_shape
        self.normalizing_constant = math.log(self.num_components)
        self.prototypes = torch.nn.Parameter(
                            torch.randn(self.num_components, *prototype_shape), requires_grad=True
                        )

    def prepare_prototypes(self):
        if "gumbel" in self.prior_type:
            prototypes = torch.nn.functional.gumbel_softmax(self.prototypes, tau=self.temperature_scaler, hard=True)
        elif "softmax" in self.prior_type:
            prototypes = torch.softmax(self.temperature_scaler * self.prototypes, 2)
        else:
            prototypes = self.prototypes
        return prototypes
    
    def kl_div(self, mu, log_var, prototypes_mu, prototypes_log_var):
        kl, kl_divs = kl_divergence_gaussian_vs_mog(mu, log_var, prototypes_mu, prototypes_log_var)
        self.kl_divs = kl_divs

        #kl_divs_max, _ = torch.max(kl_divs, 1)  # MB x 1

        #kl = self.normalizing_constant - torch.logsumexp(- kl_divs, dim=1)
        return kl


class MogPrior(torch.nn.Module):
    def __init__(self, z_dim, num_components, prior_type, init_scaler=1):
        super(MogPrior, self).__init__()
        self.num_components = num_components
        self.prior_type = prior_type
        self.z_dim = z_dim
        self.prior_init_scaler = init_scaler
        print(self.prior_init_scaler)
        if self.num_components == 1:
            #self.prior_means =  
            self.register_buffer("prior_means", torch.zeros(1, self.z_dim, self.num_components))
            self.register_buffer("prior_rhos", (math.log(math.e - 1)) * torch.ones(1, self.z_dim, self.num_components))

        else:
            means = torch.randn(1, self.z_dim, self.num_components)
            for m in range(self.num_components):
                means[0,:,m] = m
            self.register_buffer("prior_means", means )
            variances = self.prior_init_scaler * torch.ones(1, self.z_dim, self.num_components)
            #variances =  variances
            print(self.prior_init_scaler)
            self.register_buffer("prior_rhos", var_to_rho(variances))
            
            if (self.num_components < self.z_dim):
                #self.prior_means = torch.linalg.qr(self.prior_means).Q
                print("Prior mean orthogonality:", (self.prior_means[0].T @ self.prior_means[0] - torch.eye(self.num_components)).norm(2))

            if not ("fixmean" in self.prior_type):
                self.prior_means =  torch.nn.Parameter(self.prior_means, requires_grad=True)
            
            if not ("fixvar" in self.prior_type):
                self.prior_rhos = torch.nn.Parameter(self.prior_rhos, requires_grad=True)         

    def kl_div(self,mu,log_var):
        kl, kl_divs = kl_divergence_gaussian_vs_mog(mu, log_var, self.prior_means, rho_to_logvar(self.prior_rhos))
        self.kl_divs = kl_divs
        return kl

    def sample(self,N):
        X = []
        for m in range(self.num_components):
            mu,log_var = self.prior_means[...,m], rho_to_logvar(self.prior_rhos)[...,m]
            X.append(torch.distributions.MultivariateNormal(loc=mu,covariance_matrix=log_var.diag()).sample(N))
        return X
    