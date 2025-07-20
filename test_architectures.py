from spn import SPN, Layer
from variable_layer import *
from gmm_layer import *

def iris_nll_gmm(device):
    n_variables = 5
    spn = SPN(device)

    leaf_layer = VariableLayer(n_variables=n_variables, n_var_nodes=5) 
    spn.addLayer(leaf_layer)

    g1 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=20)
    spn.addLayer(g1)
    
    g2 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=8)
    spn.addLayer(g2)

    root = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=1)
    root.type = TYPE_GPRODUCT_ROOT
    spn.addLayer(root)
    return spn.to(device)

def iris_nll_gmm_bernoulli(device):
    n_variables = 7
    spn = SPN(device)

    leaf_layer = VariableLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    spn.addLayer(leaf_layer)

    g1 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=20)
    spn.addLayer(g1)
    
    g2 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=8)
    spn.addLayer(g2)

    root = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=1)
    root.type = TYPE_GPRODUCT_ROOT
    spn.addLayer(root)
    return spn.to(device)

# TODO test on adult income dataset
def uciml_nll(device):
    n_cat = 8
    n_gauss = 5
    n_variables = n_cat + n_gauss + 1
    spn = SPN(device)

    leaf_layer = VariableLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    spn.addLayer(leaf_layer)

    g1 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=128)
    spn.addLayer(g1)
    
    g2 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=64)
    spn.addLayer(g2)

    g3 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=32)
    spn.addLayer(g3)

    root = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=1)
    root.type = TYPE_GPRODUCT_ROOT
    spn.addLayer(root)
    return spn.to(device)

def mnist_spn_cross_entropy(device):
    n_variables = 28*28
    spn = SPN(device)

    leaf_layer = VariableLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    spn.addLayer(leaf_layer)   

    g1 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=128, dropout=0.5)
    spn.addLayer(g1)

    g2 = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=64, dropout=0.0)
    spn.addLayer(g2)
 
    root = GMixture(prev_layer=spn.layers[-1], n_prod_nodes=10)
    spn.addLayer(root)
    return spn.to(device)
