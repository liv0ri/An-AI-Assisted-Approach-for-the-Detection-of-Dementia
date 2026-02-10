import torch
import torch.nn as nn 

class LayerNorm(nn.Module): 
    def __init__(self, dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim)) # gamma is a learnable scaling parameter, initialized to ones. 
        self.beta = nn.Parameter(torch.zeros(dim)) # beta is a learnable shifting parameter, initialized to zeros. 
        self.eps = eps # Stores the epsilon value.

    def forward(self, x): 
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps) # Subtracts the mean and divides by the standard deviation for normalization.
        out = self.gamma * out + self.beta # Applies the learnable scaling and shifting  parameters to the normalized output.

        return out 