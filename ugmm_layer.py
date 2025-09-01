import torch
from torch import nn
from model_defs import *

from torch import distributions as dist

from ugmm_net import Layer

class uGMMLayer(Layer): 
    def __init__(self, prev_layer, n_ugmm_nodes, dropout=0.0):
        super(uGMMLayer, self).__init__(dropout)
        self.type = TYPE_UGMM_LAYER
        self.prev_layer = prev_layer
        self.n_nodes = n_ugmm_nodes
        self.n_input_variables = prev_layer.n_nodes 
        self.dropout = dropout

        self.mu = nn.Parameter(torch.randn(self.n_nodes, self.n_input_variables))
        self.std = nn.Parameter(torch.log(torch.rand(self.n_nodes, self.n_input_variables)))
        self.k = nn.Parameter(torch.rand(self.n_nodes, self.n_input_variables))

    def log_prob_prior(self):
         x = self.prev_layer.output
         # TODO define prior 
         return x 
   
    def forward(self, training):
        x = self.prev_layer.output
        batch_size = x.size(0)         
        x = x.unsqueeze(1)
        std = self.std.exp()
        k = self.k
        if training and self.dropout > 0.0:
            mask = self._dropout(k)
            k = k + mask
        x = -(x - self.mu)**2 / (2*std**2) - torch.log(std) - torch.log(torch.sqrt(2 * torch.tensor(3.14159265358979323846)))              
        x = x + k.log_softmax(dim=1)
        x = torch.logsumexp(x, dim=2)
        self.output = x
        return x              

    def forward_mpe(self):
        x = self.prev_layer.output
        batch_size = x.size(0)         
        x = x.unsqueeze(2)
        std = self.std.exp()
        x = -(x - self.mu)**2 / (2*std**2) - torch.log(std) - torch.log(torch.sqrt(2 * torch.tensor(3.14159265358979323846)))              
        x = x + self.k.log_softmax(dim=1)
        x = torch.logsumexp(x, dim=3, keepdim=False)
        self.output = x
        return x     