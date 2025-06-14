# define class for scale mixture gaussian prior


import torch
import math
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaleMixtureGaussian:                               
    def __init__(self, mixture_weight, stddev_1, stddev_2):
        super().__init__()
        # mixture_weight is the weight for the first gaussian
        self.mixture_weight = mixture_weight
        # stddev_1 and stddev_2 are the standard deviations for the two gaussians
        self.stddev_1 = stddev_1
        self.stddev_2 = stddev_2
        # create two normal distributions with the specified standard deviations
        self.gaussian1 = torch.distributions.Normal(0,stddev_1)
        self.gaussian2 = torch.distributions.Normal(0,stddev_2)


    def log_prob(self, x):
        prob1 = torch.exp(self.gaussian1.log_prob(x))
        prob2 = torch.exp(self.gaussian2.log_prob(x))
        return (torch.log(self.mixture_weight * prob1 + (1-self.mixture_weight) * prob2)).sum()
    
# define class for gaussian node
class GaussianNode:
    def __init__(self, mean, rho_param):
        super().__init__()
        self.mean = mean
        self.rho_param = rho_param
        self.normal = torch.distributions.Normal(0,1)
    
    # Calculate the standard deviation from the rho parameter
    def sigma(self):
        return torch.log1p(torch.exp(self.rho_param))

    # Sample from the Gaussian node
    def sample(self):
        epsilon = self.normal.sample(self.rho_param.size()).cuda()
        return self.mean + self.sigma() * epsilon
    
    # Calculate the KL divergence between the prior and the variational posterior
    def log_prob(self, x):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma()) - ((x - self.mean) ** 2) / (2 * self.sigma() ** 2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, mu_init, rho_init, prior_init):
        super().__init__()

        # Initialize the parameters for the weights and biases
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features).uniform_(*mu_init))
        self.weight_rho_param = nn.Parameter(torch.empty(out_features, in_features).uniform_(*rho_init))
        self.weight = GaussianNode(self.weight_mean, self.weight_rho_param)

        self.bias_mean = nn.Parameter(torch.empty(out_features).uniform_(*mu_init))
        self.bias_rho_param = nn.Parameter(torch.empty(out_features).uniform_(*rho_init))
        self.bias = GaussianNode(self.bias_mean, self.bias_rho_param)
        
        self.weight_prior = ScaleMixtureGaussian(prior_init[0], math.exp(prior_init[1]), math.exp(prior_init[2]))
        self.bias_prior = ScaleMixtureGaussian(prior_init[0], math.exp(prior_init[1]), math.exp(prior_init[2]))

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        weight = self.weight.sample()
        bias = self.bias.sample()

        return nn.functional.linear(x, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.experiment = model_params['experiment']
        self.mu_init = model_params['mu_init']
        self.rho_init = model_params['rho_init']
        self.prior_init = model_params['prior_init']

        self.fc1 = BayesianLinear(self.input_shape, self.hidden_units, self.mu_init, self.rho_init, self.prior_init)
        self.fc1_activation = nn.ReLU()
        self.fc2 = BayesianLinear(self.hidden_units, self.hidden_units, self.mu_init, self.rho_init, self.prior_init)
        self.fc2_activation = nn.ReLU()
        self.fc3 = BayesianLinear(self.hidden_units, self.classes, self.mu_init, self.rho_init, self.prior_init)
    
    def forward(self, x):
        if self.experiment == 'classification':
            x = x.view(-1, self.input_shape) # Flatten images
        x = self.fc1_activation(self.fc1(x))
        x = self.fc2_activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def log_prior(self):
        return self.fc1.log_prior + self.fc2.log_prior + self.fc3.log_prior
    
    def log_variational_posterior(self):
        return self.fc1.log_variational_posterior + self.fc2.log_variational_posterior + self.fc3.log_variational_posterior


    def get_nll(self, outputs, target, sigma=1.):
        if self.experiment == 'regression': #  -(.5 * (target - outputs) ** 2).sum()
            nll = -torch.distributions.Normal(outputs, sigma).log_prob(target).sum()
        elif self.experiment == 'classification':
            nll = nn.CrossEntropyLoss(reduction='sum')(outputs, target)
        return nll

    def sample_elbo(self, x, target, beta, samples, sigma=1.):
        log_prior = torch.zeros(1).to(device)
        log_variational_posterior = torch.zeros(1).to(device)
        negative_log_likelihood = torch.zeros(1).to(device)

        for i in range(samples):
            output = self.forward(x)
            log_prior += self.log_prior()
            log_variational_posterior += self.log_variational_posterior()
            negative_log_likelihood += self.get_nll(output, target, sigma)

        log_prior = beta*(log_prior / samples)
        log_variational_posterior = beta*(log_variational_posterior / samples) 
        negative_log_likelihood = negative_log_likelihood / samples
        loss = log_variational_posterior - log_prior + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood    

class MLP(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.experiment = model_params['experiment']

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.classes))
    
    def forward(self, x):
        if self.experiment == 'classification':
            x = x.view(-1, self.input_shape) # Flatten images
        
        x = self.net(x)
        return x

class MLP_Dropout(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.experiment = model_params['experiment']

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_units, self.classes))
    
    def forward(self, x):
        if self.experiment == 'classification':
            x = x.view(-1, self.input_shape) # Flatten images
       
        x = self.net(x)
        return x