import torch
from torch import nn
from model_defs import *

from ugmm_net import Layer

class InputLayer(Layer): 
    def __init__(self, n_variables, n_var_nodes, dropout=0.0):
        super(InputLayer, self).__init__(dropout)
        self.type = TYPE_INPUT_LAYER
        self.id = 0
        self.n_nodes = n_var_nodes
        self.n_input_variables = n_variables
        self.dropout = dropout

    def forward(self, x, training):
        self.observed = x
        self.output = x
        return x    

    # Perform MPE for a single variable (bruteforce)
    def forward_mpe(self, x, mpe_vars, mpe_states):
        self.mpe_observed = x
        batch_size = x.size(0)         
        n_states = len(mpe_states)
        x = x.expand(n_states, batch_size, x.shape[-1])
        y = x.clone()
        for i in range(0, n_states):
            y[i,:, mpe_vars[0]] = mpe_states[i]

        self.output = y
        return x    