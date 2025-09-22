from ugmm_net import *
from input_layer import *
from ugmm_layer import *

def iris_nll_gmm(device):
    n_variables = 5
    model = uGMMNet(device)

    input_layer = InputLayer(n_variables=n_variables, n_var_nodes=5) 
    model.addLayer(input_layer)

    g1 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=10)
    model.addLayer(g1)
    
    g2 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=5)
    model.addLayer(g2)

    root = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=1)
    model.addLayer(root)
    return model.to(device)

def iris_nll_gmm_bernoulli(device):
    n_variables = 7
    model = uGMMNet(device)

    input_layer = InputLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    model.addLayer(input_layer)

    g1 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=20)
    model.addLayer(g1)
    
    g2 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=8)
    model.addLayer(g2)

    root = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=1)
    model.addLayer(root)
    return model.to(device)

def mnist_cross_entropy(device):
    n_variables = 28*28
    model = uGMMNet(device)

    input_layer = InputLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    model.addLayer(input_layer)   

    g1 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=128, dropout=0.3)
    model.addLayer(g1)

    g2 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=64, dropout=0.0)
    model.addLayer(g2)
 
    root = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=10, dropout=0.0)
    model.addLayer(root)
    return model.to(device)
