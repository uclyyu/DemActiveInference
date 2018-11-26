## What is this?
Variational Bayesian inversion of dynamical system, whcih also includes Active Inference, built upon Theano.

## Reference
Codes in this repository implement the following work:
1. KJ Friston, N Trujillo-Barreto, and J Daunizeau (2008) "DEM: A variational treatment of dynamic systems."; NeuroImage.
2. KJ Friston, KE Stephan, B Li, and J Daunizeau (2010) "Generalised filtering."; Mathematical Problems in Engineering.
3. KJ Friston, J Daunizeau, and SJ Kiebel (2009) "Reinforcement learning or Active Inference?"; PLoS ONE.

## How to use
```python
import vid

# create an empty model
hdm = vid.HierarchicalDynamicModel()

# add one layer whose functional form is linear convolution
vid.DemoLinearConvolutionModule(hdm);

# add prior over causal state
vid.UnivariateGMModule(hdm);

# use DEM inversion scheme
dem = vid.DynamicExpectationMaximisation(hdm)
# initialise computation graph and compile
dem.initialise()

```
