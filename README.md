# Scalable Bayesian Optimization


## TODO:

- [ ] Implement vanilla GPy
- [ ] Implement RFF
  - [ ] Use chol for inversion
  - [ ] Consider computing std instead of covar in `get_statistics`
  - [ ] Consider whether inversion is right
  - [ ] Allow for Hyperparameter optimization
  - [ ] Verify that p(w) is a distribution
  - [ ] Convert `embed` to pytorch?
  - [ ] Problematic to compute $k(x,X)$ using non-approximate 
  kernel?
  - [ ] Test on multi-dim
- [ ] Annotate numpy with dimensions (somehow!)
- [ ] Allow for debugging in vscode jupyter
- [ ] Test on benchmark
- [ ] Implement QFF with SE
- [ ] Implement RFF with Matern
- [ ] Extend to multi-objective
- [ ] Optimization of Acquisition function that is not fixed (from QFF)
- [ ] Unbounded optimization
- [ ] Additive model (split into independent BO problems)
- [ ] Find additive structure with Gibbs sampling
- [ ] Support additive models with overlapping groups through message passing

- [ ] Implement KISS-GP model using gpytorch


## Installation


- General requirements:
  ```bash
  source $HOME/miniconda/bin/activate
  conda create -n lions python=3.6
  source activate lions
  conda install -y pytorch torchvision -c pytorch
  conda install -y gpytorch -c gpytorch
  conda install -y -c conda-forge numpy blas seaborn scipy matplotlib pandas gpy
  pip install pydot-ng
  pip install gpyopt

  conda install -y scikit-learn
  pip install git+https://github.com/IDSIA/sacred.git
  pip install pyyaml GitPython pymongo
  pip install incense

  git clone https://github.com/jmetzen/gp_extras.git
  cd gp_extras
  python setup.py install 
  ```

- Notebook requirements:
  ```bash
  conda -y install notebook
  pip install ipywidgets
  jupyter nbextension enable --py widgetsnbextension
  ```

- Plotly requirements for jupyterlab:  
  ```bash
  jupyter labextension install @jupyterlab/plotly-extension
  ```

## Omniboard

```
npm install -g omniboard
omniboard -m localhost:27017:lions
```

## Troubleshoot

  Temporarily fixed `conda install numpy=1.15.4` (https://github.com/SheffieldML/GPy/issues/728#issuecomment-459379268) to fix plotting problem in GPy.
