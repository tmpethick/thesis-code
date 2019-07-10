# Scalable Bayesian Optimization

## Datasets

https://arxiv.org/pdf/1807.02125.pdf
https://archive.ics.uci.edu/ml/datasets.php

```
challenger
fertility
slump
automobile
servo
cancer
hardware
yacht
autompg
housing
forest
stock
energy
concrete
solar
wine
skillcraft
pumadyn
elevators
kin40k
keggu
3droad
electric
```


## Installation

On linux for pytorch:
```
conda install -y -c mkl
conda install -y pytorch-cpu torchvision-cpu -c pytorch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

- General requirements:
```bash
source $HOME/miniconda/bin/activate
conda create -n lions python=3.6
source activate lions
conda install -y pytorch torchvision -c pytorch
pip install gpytorch=0.3.3
conda install -y -c conda-forge numpy blas seaborn scipy matplotlib pandas gpy
conda install -y pylint
pip install pydot-ng
pip install gpyopt
pip install emcee

conda install -y scikit-learn==0.20.3
pip install git+https://github.com/baldassarreFe/sacred.git@gh-issue-442
pip install dnspython
pip install pyyaml GitPython pymongo
pip install incense

pip install cvxopt

git clone https://github.com/jmetzen/gp_extras.git
cd gp_extras
python setup.py install 

pip install mpi4py
```

- Notebook requirements:
  ```bash
  conda -y install notebook
  pip install ipywidgets
  jupyter nbextension enable --py widgetsnbextension

  pip install addict
  pip install jupyter_contrib_nbextensions
  jupyter contrib nbextension install --user
  pip install jupyter_nbextensions_configurator
  jupyter nbextensions_configurator enable --user
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

For debugging:
```
sbatch hpc.sh 'python' 'runner.py' 'print_config' 'with' 'obj_func={"name": "Sinc"}'
```

## Adaptive Sparse Grid installation

- Run:
  ```
cd SparseGridCode/TasmanianSparseGrids
make
cd ../
cd pyipopt
./install.sh
echo " IPOPT and PYIPOPT is installed "
  ```

Note: We replaced `basestring` with `str` in `SparseGridCode/TasmanianSparseGrids/InterfacePython/TasmanianSG.py` to make the library python3 compatible.


## Profiling

```
pip install memory_profiler
sudo mprof run --include-children python debug_notebook.py
mprof plot --backend TkAgg
```

```
pip install pympler
```

```python
from pympler import muppy, summary
all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
summary.print_(sum1)
```

## DTU HPC

```
ssh s144448@login2.hpc.dtu.dk
bqueues
bstat
```


## Troubleshoot

  Temporarily fixed `conda install numpy=1.15.4` (https://github.com/SheffieldML/GPy/issues/728#issuecomment-459379268) to fix plotting problem in GPy.
