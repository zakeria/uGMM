<p align="center">
  <img src="https://github.com/zakeria/uGMM/raw/main/images/logo.png" alt="uGMM Logo" width="300"/>
</p>

## uGMM-NN: Univariate Gaussian Mixture Model Neural Network

This repository introduces the **Univariate Gaussian Mixture Model Neural Network Model (uGMM-NN)**. This experimental feedforward neural network architecture replaces traditional neuron operations with **probabilistic univariate Gaussian mixture nodes**. By parameterizing neurons with means, variances, and mixing coefficients, uGMM-NNs capture multimodality and uncertainty that standard MLP neurons cannot represent. This mixture-based view allows richer probabilistic reasoning within deep neural networks, making them especially promising as building blocks for next-generation architectures.


### Model Formulation

The uGMM-NN reimagines the fundamental building block of a feedforward neural network. Instead of a neuron computing a weighted sum of inputs and applying a fixed non-linear activation, each "neuron" in a uGMM-NN is a **univariate Gaussian Mixture Model (uGMM)**.

### Univariate GMM Nodes

A uGMM neuron j receives N inputs (x₁, ..., xₙ) from the previous layer. 
Its associated Gaussian Mixture Model has exactly N components, 
each corresponding to one input. The means (μⱼ,ₖ), variances (σ²ⱼ,ₖ), and mixing coefficients (πⱼ,ₖ) are learnable parameters unique to neuron j.

<p align="center">
  <img src="https://github.com/zakeria/uGMM/raw/main/images/model_architecture.png" alt="example model architecture" width="580"/>
</p>

---

### Example Usage

The uGMM-NN follows a classic feedforward neural network architecture, comprising input, hidden, and output layers. Each neuron in the network represents a univariate Gaussian mixture model (uGMM), where the mixture components correspond to inputs from the previous layer. Conceptually, the model forms a hierarchical composition of uGMMs, enabling the construction of complex, high-dimensional probability distributions through successive transformations.

#### Define the model
Instead of adding dense layers, we stack univariate Gaussian Mixture layers (uGMM) that represent mixtures over inputs from the previous layer:
```python
def mnist_fc_ugmm(device):
    n_variables = 28*28
    model = uGMMNet(device)

    # Input layer with one variable node per pixel
    input_layer = InputLayer(n_variables=n_variables, n_var_nodes=n_variables) 
    model.addLayer(input_layer)   

    # First hidden layer with 128 uGMM nodes + dropout
    g1 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=128, dropout=0.5)
    model.addLayer(g1)

    # Second hidden layer with 64 uGMM nodes
    g2 = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=64)
    model.addLayer(g2)
 
    # Output layer with 10 uGMM nodes (for MNIST classes)
    root = uGMMLayer(prev_layer=model.layers[-1], n_ugmm_nodes=10)
    model.addLayer(root)

    return model.to(device) 
```
#### Train the model
Training a uGMM-NN looks almost identical to training a standard FFNN model. You define an optimizer and a loss function, then run a forward–backward pass loop:
```python
model = mnist_fc_ugmm(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()    

num_epochs = 100
for epoch in range(num_epochs):
    for batch_index, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  

        # Flatten MNIST images into vectors
        batch_size = inputs.shape[0]
        data = inputs.reshape(batch_size, 28*28).to(device)

        # Forward pass with uGMM inference
        output = model.infer(data, training=True)
        loss = criterion(output, labels.to(device))         

        # Backpropagation
        loss.backward()  
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

```
The `notebooks` directory contains Jupyter notebooks that demonstrate the usage of this library with complete examples.

- [Example (discriminative) inference on the MNIST dataset](./notebooks/mnist_dataset.ipynb)
- [Example (generative) inference on the Iris dataset using a uGMM trained with NLL loss.](./notebooks/iris_dataset.ipynb)
---

### Citation
For details on uGMM-NN, see [the paper](https://arxiv.org/abs/2509.07569), and to cite it, use:

```bibtex
@article{Zakeria2025uGMM,
  author    = {Zakeria Sharif Ali},
  title     = {uGMM-NN: Univariate Gaussian Mixture Model Neural Network},
  journal   = {arXiv preprint arXiv:2509.07569},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.07569}
}}
