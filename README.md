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

On linux for pytorch:
conda install -y -c anaconda mkl
conda install -y pytorch-cpu torchvision-cpu -c pytorch

- General requirements:
```bash
source $HOME/miniconda/bin/activate
conda create -n lions python=3.6
source activate lions
conda install -y pytorch torchvision -c pytorch
pip install gpytorch
conda install -y -c conda-forge numpy blas seaborn scipy matplotlib pandas gpy
conda install -y pylint
pip install pydot-ng
pip install gpyopt
pip install emcee

conda install -y scikit-learn
pip install git+https://github.com/IDSIA/sacred.git
pip install dnspython
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
omniboard --mu "mongodb+srv://admin:<password>@lions-rbvzc.mongodb.net/test?retryWrites=true"
PASS=$(python -c 'from src import env; print(env.MONGO_DB_PASSWORD)'); omniboard --mu "mongodb+srv://admin:${PASS}@lions-rbvzc.mongodb.net/test?retryWrites=true"
```

If MongoDB is living on a firewalled server (not currently the case):
```
ssh -fN -l root -i path/to/id_rsa -L 9999:localhost:27017 host.com
ssh -N -f -L localhost:8889:localhost:7000 rkarimi@simba-compute-gpu-3
```

## Jupyter notebook

```
jupyter notebook
```

## Server

It requires to have `simba` configured in `~/.ssh/config`.

## Troubleshoot

  Temporarily fixed `conda install numpy=1.15.4` (https://github.com/SheffieldML/GPy/issues/728#issuecomment-459379268) to fix plotting problem in GPy.
