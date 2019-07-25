# Scalable Bayesian Optimization

This repository includes the code for my master thesis entitled "Scalable Gaussian Processes for Economic Models".

There are three ways to dig into the repository:

* **[A Demo Notebook](demo.ipynb)**: which illustrates how to run an experiment locally or on a High Performance Computing (HPC) environment and how to aggregate and inspect those results afterwards.
* **[A Results Notebook](results.ipynb)**: to inspect and reproduce results and plots from the thesis. All experiments on which the thesis is based is stored in a publicly accessible MongoDB database from which this notebook ensembles the figures and tables.
- **[The thesis](thesis.pdf)**: which describes the model and results.

_Note: we strong recommend using the Table of Content Jupyter notebook extension to navigate the files._


## Workflow

There are roughly three steps to executing an experiment.

- Define a experiments as a JSON serializable Python dictionary which specifies the _model_ and _environment_.
  ```python
  execute({
    'tag': 'demo',
    'obj_func': {'name': 'Sinc'},
    'model': {'name': 'GPModel',
              'kwargs': {'learning_rate': 0.1}},
  })
  ```
- The experiment is executed on a HPC environment and collected in a centralized MongoDB database.
- Inspect the results on the MongoDB database in a notebook as a Pandas DataFrame.
  ```python
  get_df(**{'config.tag': 'demo'})
  ```


## Code Outline

The code (i.e. everything in `src/`) is roughly divided into four parts:

- **Experiment**: This is where most of the thesis specific code resides. It includes a `Runner` which defines how test and training data is drawn and what plots to generate.
- **Models**: Various models that fits to particular training example and then predict based on this. Most are probabilistic and yields a predictive mean and variance for test locations.
- **Environments**: The environment from which the models should train and test on. This can either be _synthetic functions_ for which the ground truth is known, _simulations_ where a point evaluation is generated on the fly, and _data set_ with fixed evaluation locations.


### Environments

The following provides a high-level overview of the available environments:

- **Non-stationary**: Sinusoidals with varying amplitude and length-scale.
- **Discontinous**: Step functions and kinks.
- **Financial/Economic**: simulated models such as the growth model and option pricing as well as a cleaned stock marked data sets.
- **UCI**: Various (normalized) machine learning data ported from [https://people.orie.cornell.edu/andrew/code/](https://people.orie.cornell.edu/andrew/code/).
- **Natural Sound and Precipitation**: Datasets of low dimensions with many observations ported from [https://github.com/kd383/GPML_SLD](https://github.com/kd383/GPML_SLD).
- **Genz 1984**: For integrands scalable to arbitrary dimensionality.
- **Optimization benchmarks**: Synthetic functions such as Branin and Rosenbrock ported from GPyOpt.
- **Helpers**: For automatically normalizing and creating embeddings.


## Installation

```
conda create -n sgp python=3.6
source activate sgp
conda env update -f environment.yml
```

(Note: We create the environment before populating it because of a conda bug where Python 3.6 is otherwise not accessible during installation.)

For Sparse Grid requirements see further down.


### Optional installations

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

  We recommend enabling the `table of content` plugin from `jupyter_contrib_nbextensions` to navigate the included notebooks.
- Growth model
  ```
  source activate sgp
  source setup_env.sh
  sh install_growth.sh
  ```
- Adaptive Sparse Grid installation
  ```
  cd SparseGridCode/TasmanianSparseGrids
  make
  cd ../pyipopt
  ./install.sh
  echo " IPOPT and PYIPOPT is installed "
  ```
  
  Note: We replaced `basestring` with `str` in `SparseGridCode/TasmanianSparseGrids/InterfacePython/TasmanianSG.py` to make the library python3 compatible.
- For `LocalLengthScaleGPModel`
  ```
  git clone https://github.com/jmetzen/gp_extras.git
  cd gp_extras
  python setup.py install 
  ```
- Pytorch On GPU enabled linux machine:
  ```
  source $HOME/miniconda/bin/activate
  conda install -y pytorch-cpu torchvision-cpu -c pytorch
  conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
  ```


## Omniboard

To view the MongoDB records of the experiments in a browser interface run the following:

```
npm install -g omniboard
omniboard -m localhost:27017:lions
omniboard --mu "mongodb+srv://admin:<password>@lions-rbvzc.mongodb.net/test?retryWrites=true"
PASS=$(python -c 'from src import env; print(env.MONGO_DB_PASSWORD)'); omniboard --mu "mongodb+srv://admin:${PASS}@lions-rbvzc.mongodb.net/test?retryWrites=true"
```

Remember to replace `<password>`.

If MongoDB is living on a firewalled server (not currently the case):
```
ssh -fN -l root -i path/to/id_rsa -L 9999:localhost:27017 host.com
ssh -N -f -L localhost:8889:localhost:7000 user@server
```

## Jupyter notebook

```
jupyter notebook
```

## HPC environments

Currently we support EPFL and DTUs.

### EPFL HPC

It requires to have `simba` configured in `~/.ssh/config`.
```
Host simba.epfl.ch simba simba-fe
     Hostname simba.epfl.ch
     User <username>
     ForwardAgent yes
     ForwardX11 yes
     ForwardX11Timeout 596h
     DynamicForward 3333
Host simba-compute-01 simba-compute-02 simba-compute-03 simba-compute-04 simba-compute-05 simba-compute-06 simba-compute-07 simba-compute-08 simba-compute-09 simba-compute-10 simba-compute-11 simba-compute-12 simba-compute-13 simba-compute-14 simba-compute-15 simba-compute-16 simba-compute-17 simba-compute-18 simba-compute-gpu-1 simba-compute-gpu-2 simba-compute-gpu-3
    User <username>
    ForwardAgent yes
    ForwardX11 yes
    ForwardX11Timeout 596h
    DynamicForward 3333
    ServerAliveInterval    60
    TCPKeepAlive           yes
    ProxyJump              simba
Host *
    XAuthLocation /opt/X11/bin/xauth
```

Remember to replace `<username>`.

For debugging purposes you can submit a no-op script directly from the server:
```
ssh simba
sbatch path/to/hpc.sh 'python' 'runner.py' 'print_config' 'with' 'obj_func={"name": "Sinc"}'
```

### DTU HPC

```
ssh <username>@login2.hpc.dtu.dk
```

Available commands:

* **`bqueues`** list available server queues.
* **`bstat`** list jobs.
* **`qrsh`** run interactive job server.

Trick to view plots on the server:
```bash
ssh -Y <username>@login2.hpc.dtu.dk
eog path/to/file.png
```

## Profiling notes

To plot the memory use:
```bash
pip install memory_profiler
sudo mprof run --include-children python debug_notebook.py
mprof plot --backend TkAgg
```

To print memory use pr. object type (Note that this requires modifying the source code):

```bash
pip install pympler
```

```python
from pympler import muppy, summary
all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
summary.print_(sum1)
```
