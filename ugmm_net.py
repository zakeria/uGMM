import torch
from torch import nn

class uGMMNet(nn.Module):
    def __init__(self, device, dropout=0.0):
        super(uGMMNet, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        self.dropout = dropout
    
    def addLayer(self, layer):
        layer.device = self.device
        self.layers.append(layer)

    def infer(self, x, training=False):
        n_layers = len(self.layers)
        leaf_layer = self.layers[0]
        y = leaf_layer.forward(x, training)
        for i in range(1, n_layers):
            layer = self.layers[i]
            y = layer.eval(training)        
        return y

    def infer_mpe(self, x, mpe_vars, mpe_states):
        n_layers = len(self.layers)
        leaf_layer = self.layers[0]
        y = leaf_layer.forward_mpe(x, mpe_vars, mpe_states)
        for i in range(1, n_layers):
            layer = self.layers[i]
            y = layer.eval_mpe()    
        return y

class Layer(nn.Module):
    def __init__(self, dropout=0.0):
        super(Layer, self).__init__()
        self.dropout = dropout
        self._bernoulli_dist = torch.distributions.Bernoulli(probs=self.dropout)

    def _dropout(self, x: torch.Tensor):
        dropout_indices = self._bernoulli_dist.sample(x.shape,).float()
        dropout_indices[dropout_indices == 1] = -torch.inf
        return dropout_indices.to(self.device)

    def addNode(self, node):
        self.nodes.append(node)
    
    def eval(self, training):
        return self.forward(training)
   
    def eval_mpe(self):
        return self.forward_mpe()
 