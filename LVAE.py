import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LadderVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, d_dim_vector, z_dim_vector):
        """
        personal implementation of ladder variational auoencoder
        :param input_dim: size of input; for mnist it will be 28*28
        :param d_dim_vector: a vector with elements determine the size of MLP e.g [16, 32]
        :param z_dim_vector: a vector in whcih each element determine the z size e.g [16, 32]
        """
        super(LadderVariationalAutoencoder, self).__init__()
        self.d_dim_v = d_dim_vector
        self.z_dim_v = z_dim_vector
        self.input_dim = input_dim
        self.num_of_layer = len(z_dim_vector)

        # encoder
        self.encoder = nn.ModuleList([EncoderBlock(layer[0], layer[1], layer[2]) for layer in
                        zip([input_dim]+d_dim_vector[:-1], d_dim_vector, z_dim_vector)])
        # decoder
        self.decoder = [Decoder_first_layer()]
        self.decoder = nn.ModuleList(self.decoder+[DecoderBlock(layer[0], layer[1], layer[2]) for layer in
                             zip(list(reversed(z_dim_vector))[:-1], list(reversed(d_dim_vector))[:-1], list(reversed(z_dim_vector))[1:])])
        # reconstructed output
        # self.reconst1 = nn.Linear(self.z_dim_v[0], self.input_dim)
        self.reconst = nn.Sequential(nn.Linear(self.z_dim_v[0], self.input_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid())

    def forward(self, x):
        d = x
        encoder_paameter = []
        latents = []

        # top-down
        for layer in self.encoder:
            d, mu, sigma = layer(d)
            encoder_paameter.append((mu, sigma))

        p_parameters = []
        q_parameters = []
        encoder_parameter = list(reversed(encoder_paameter))
        z = None

        # bottom-up
        for layer_index, layer in enumerate(self.decoder):
            if layer_index ==0:
                mu_t, sigma_t, mu_merged, sigma_merged = layer(*encoder_parameter[layer_index])
            else: #note we need to sample from q(z|x)
                mu_t, sigma_t, mu_merged, sigma_merged = layer(*encoder_parameter[layer_index], z)
            z = reparametrization_trick(mu_merged, sigma_merged)
            p_parameters.append((mu_t, sigma_t)) # note we used parameter sharing; mu_t = mu_p and sigma_t = sigma_p
            q_parameters.append((mu_merged, sigma_merged))
            latents.append(z)
        x_reconst = self.reconst(z)  # ????
        # x = F.sigmoid(z)
        return x_reconst, latents, encoder_paameter, p_parameters, q_parameters





def reparametrization_trick(mu, var):
    '''
    Function that given the mean (mu) and the  (var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample
    :param mu: mean of the z_variables
    :param sigma: variance of the latent variables (as in the paper)
    :return: z = mu + sigma * noise
    '''
    # compute the standard deviation from the variance
    std = torch.sqrt(var+1e-8)

    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)
    # return mu

def LVAE_optimizer(prediction, target, latents, encoder_parameters, decoder_parameters, target_type= "MSE"):
    re_loss = reconst_loss(prediction, target, target_type)
    kl = kl_divergence(latents, encoder_parameters, decoder_parameters)
    negative_elbo = re_loss + kl
    return negative_elbo, re_loss, kl


def reconst_loss(prediction, target, target_type ):
    if target_type == "bernouli":
        loss = torch.nn.BCELoss(reduction ="sum") #???
    else:
        loss = torch.nn.MSELoss(reduction ="sum")
    return loss(prediction, target)

def gaussian_log_pdf( x , mean , variance):
    """
    this function is used to calculate the logarithm density of gaussian with given parameters at x.
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    x: given point
    mean: mean of gaussian distribution
    variance: variance of gaussian distribution
    """
    k = x.shape[1]
    log_pdf = -(k/2)*math.log(math.pi*2) -.5*torch.log(torch.prod(variance,dim=1)+1e-8)-0.5*torch.sum(((x-mean)**2)* (1/(variance+1e-8)),1)
    return log_pdf

def kl_divergence(latens, p_z, q_z ):
    kl = 0
    for p_param, q_param , z in zip(p_z, q_z, latens):
        log_p = gaussian_log_pdf(z, p_param[0], p_param[1])
        log_q = gaussian_log_pdf(z, q_param[0], q_param[1])
        kl += torch.mean(log_q - log_p)
    return kl



def init():
    pass


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, d_dim, z_dim, activation= F.leaky_relu):
        super(EncoderBlock, self).__init__()
        self.shared_layer = MLP(in_dim, d_dim) # two Layer MLP eq(10)
        self.mu_hat = nn.Linear(d_dim, z_dim)  # single layer MLP eq(11)
        self.sigma_hat = nn.Linear(d_dim, z_dim)  # single layer MLP eq(12)

    def forward(self, d):
        d = self.shared_layer(d)
        mu = self.mu_hat(d)
        sigma = F.softplus(self.sigma_hat(d))
        return d, mu, sigma


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, d_dim, z_dim):
        super(DecoderBlock, self).__init__()
        self.shared_layer = MLP(in_dim, d_dim)  # two Layer MLP eq(10)
        self.mu_p = nn.Linear(d_dim, z_dim)  # single layer MLP eq(11)
        self.sigma_p = nn.Linear(d_dim, z_dim)  # single layer MLP eq(12)

    def forward(self, mu_posterior, sigma_posterior, z):
        d = self.shared_layer(z)
        sigma_p = F.softplus(self.sigma_p(d))
        mu_p = self.mu_p(d)

        mu_merged, sigma_merged = gaussian_merge(sigma_p, mu_p, sigma_posterior, mu_posterior)
        return mu_p, sigma_p, mu_merged, sigma_merged


class Decoder_first_layer(nn.Module):
    def init(self):
        super(Decoder_first_layer, self).__init__()

    def forward(self, mu_hat, sigma_hat):
        return torch.zeros_like(mu_hat), torch.ones_like(sigma_hat), mu_hat, sigma_hat


# combination of gaussian likelihood and prior eq (21)
def gaussian_merge(sigma1, mu1, sigma2, mu2):
    precision1 = (sigma1+1e-8) ** (-1)
    precision2 = (sigma2+1e-8) ** (-1)
    var = (precision1 + precision2) ** (-1)
    mean = (precision2 * mu2 + mu1 * precision1) / (precision1 + precision2)
    return mean, var


# two Layer MLP eq(10)
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(MLP, self).__init__()
        self.firstLayer = nn.Linear(in_dim, hidden_dim)
        self.secondLayer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        d = self.firstLayer(z)
        d = F.leaky_relu(d)
        d = self.secondLayer(d)
        d = F.leaky_relu(d)
        return d


