import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la

# Define a lambda function to calculate the logarithm of the absolute value of a tensor
logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    """
    Activation Normalization layer.
    This is a learnable, channel-wise affine transformation layer. Functionally similar to Batch Normalization,
    but its mean and variance are learnable parameters and are calculated independently for each channel, 
    independent of batch size.
    It uses data-dependent initialization, meaning it uses the first mini-batch of data to initialize 
    its scale and bias (loc) parameters.
    """
    def __init__(self, in_channel, logdet=True):
        """
        Initialization function.
        :param in_channel: Number of input channels
        :param logdet: Whether to return the log-determinant of the Jacobian, which is crucial for calculating log-likelihood
        """
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        """
        Data-dependent initialization method.
        Calculates the mean and standard deviation of each channel based on the first batch of data,
        and sets loc and scale such that the initial output has a mean of 0 and a standard deviation of 1.
        """
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        """
        Forward propagation: y = scale * (x + loc)
        """
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1) 

        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        """
        Reverse propagation: x = (y / scale) - loc
        """
        return output / self.scale - self.loc

class InvConv2dLU(nn.Module):
    """
    Invertible 1x1 Convolution layer.
    A 1x1 convolution is equivalent to a linear transformation on the channel dimension of the input.
    To make it invertible and easy to calculate the Jacobian determinant, the weight matrix W is parameterized 
    via its LU decomposition: W = P * L * U.
    P is a fixed permutation matrix.
    L is a lower triangular matrix with ones on the diagonal.
    U is an upper triangular matrix.
    This decomposition makes determinant calculation and matrix inversion very efficient.
    det(W) = det(P) * det(L) * det(U) = ±1 * 1 * product(diag(U))
    """
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)

        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        
        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)
        
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s)) # Store the sign of diagonal elements
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0])) # Used to construct the identity diagonal of L

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        """
        Forward propagation
        """
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        
        logdet = height * width * torch.sum(self.w_s)
        
        return out, logdet

    def calc_weight(self):
        """
        Reconstruct the full weight matrix W based on LU decomposition parameters
        """
        # L = (w_l * l_mask) + I
        l = self.w_l * self.l_mask + self.l_eye
        # U = (w_u * u_mask) + diag(s_sign * exp(w_s))
        s = self.s_sign * torch.exp(self.w_s)
        u = self.w_u * self.u_mask + torch.diag(s)
        
        # W = P @ L @ U
        weight = self.w_p @ l @ u
        
        # Shape of 1x1 convolution weights is [out_channels, in_channels, 1, 1]
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        """
        Reverse propagation: x = W^{-1} * y
        """
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

class ZeroConv2d(nn.Module):
    """
    A 2D convolution layer where both weights and biases are initialized to zero.
    It also has a learnable scaling factor, also initialized to zero.
    This makes the layer an identity transformation (output is 0, multiplied by scale=1) at the start of training, 
    which helps stabilize deep network training.
    """
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out

class AffineCoupling(nn.Module):
    """
    Affine Coupling Layer.
    This is one of the core components of Flow models. It splits input channels into two halves (x_a, x_b).
    x_a remains unchanged and is fed into a neural network (self.net), which outputs scaling factors s and translation factors t.
    Then x_b is affine transformed using s and t: y_b = s * x_b + t.
    The final output is the concatenation of y_a (which is x_a) and y_b.
    This operation is invertible, and its Jacobian determinant is very easy to calculate.
    """
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        """
        Reverse propagation
        """
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    """
    A Flow step, consisting of three parts:
    1. ActNorm: Activation Normalization
    2. InvConv2dLU: Invertible 1x1 Convolution
    3. AffineCoupling: Affine Coupling Layer
    """
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det = self.invconv(out)
        out, det2 = self.coupling(out)
        
        logdet = logdet + det
        if det2 is not None:
            logdet = logdet + det2
        
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

def gaussian_log_p(x, mean, log_sd):
    """
    Calculate the log probability density of x under a Gaussian distribution with given mean and log standard deviation.
    log P(x) = -0.5 * log(2π) - log_sd - 0.5 * ((x - mean)^2 / exp(2*log_sd))
    """
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

class Block(nn.Module):
    """
    A Block consists of n_flow Flow steps.
    In Glow's multi-scale architecture, each scale level consists of a Block.
    (Note: This code implements a single-scale Glow)
    """
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()
        self.flows = nn.ModuleList([Flow(in_channel, affine=affine, conv_lu=conv_lu) for _ in range(n_flow)])
        
        self.prior = ZeroConv2d(in_channel, in_channel * 2)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        out = input
        logdet = 0
        
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det
        
        return out, logdet

    def reverse(self, z, reconstruct=False):
        """
        Generate data from latent variable z
        """
        input = z
        for flow in reversed(self.flows):
            input = flow.reverse(input)
        return input

class Glow(nn.Module):
    """
    Main Glow model class.
    This implementation is a simplified, single-scale Glow model.
    A full Glow model contains multiple Blocks with squeeze and split operations between them.
    """
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.block = Block(in_channel, n_flow, affine=affine, conv_lu=conv_lu)

    def forward(self, input):
        """
        Forward propagation, used for training.
        Calculates the log-likelihood of input data.
        """
        # Pass through Block to get transformed latent variable z(out), log-determinant, and log prior probability log p(z)
        # out, logdet, log_p = self.block(input)
        out, logdet = self.block(input)

        
        # According to the change of variables formula, the log-likelihood log p(x) of input data x is:
        # log p(x) = log p(z) + log|det(dz/dx)|
        # where log|det(dz/dx)| is the accumulated logdet
        # return log_p, logdet, out
        return logdet, out

    def reverse(self, z, reconstruct=False):
        """
        Reverse propagation, used for generation.
        Generate a data sample from a latent variable z.
        """
        return self.block.reverse(z, reconstruct=reconstruct)
