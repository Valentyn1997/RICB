RepresentationBiasEstimation
==============================

Bounding Bias of Balancing Representations for Estimating Individualized Treatment Effects

The project is built with the following Python libraries:
1. [Pyro](https://pyro.ai/) - deep learning and probabilistic models (MDNs, NFs)
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

### Installations
First, one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000 --gunicorn-opts "--timeout 280"`

To access MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to the local browser http://localhost:5000.

## Experiments

The main training script is universal for different methods and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

Generic script with logging and fixed random seed is the following:
```console
PYTHONPATH=.  python3 runnables/train.py +dataset=<dataset> +repr_net=<model> exp.seed=10
```

### Representation learning methods for CATE (baselines)

#### Stage 0.
One needs to choose a model and then fill in the specific hyperparameters (they are left blank in the configs):
- [TARNet](https://arxiv.org/abs/1606.03976): `+repr_net=tarnet`
- [BNN](https://arxiv.org/abs/1605.03661): `+repr_net=bnnet`
- [CFR](https://arxiv.org/abs/1606.03976): `+repr_net=cfrnet`
- [InvTARNet](https://arxiv.org/abs/2001.04754): `+repr_net=inv_tarnet`
- [RCFR](https://arxiv.org/abs/2001.07426): `+repr_net=rcfrnet`
- [CFR-ISW](https://www.ijcai.org/proceedings/2019/0815.pdf): `+repr_net=cfrisw`
- [BWCFR](https://arxiv.org/abs/2010.12618): `+repr_net=bwcfr`

Models already have the best hyperparameters saved, for each model - dataset and different sizes of the representation. One can access them via: `+repr_net/<dataset>_hparams/<model>=<dim_repr_multiplier>` or `+model/<dataset>_hparams/<model>/<ipm_params>=<dim_repr_multiplier>` etc. To perform manual hyperparameter tuning use the flags `repr_net.tune_hparams=True`, and then see `repr_net.hparams_grid`. 

#### Stage 1.
Stage 1 models are propensity nets (src/models/prop_nets.py) and the conditional normalizing flow (src/models/msm.py). The hyperparameters were tuned together with the stage 0 models and are stored in the same YAML files. To perform manual hyperparameter tuning use the flags `prop_net_cov.tune_hparams=True`, `prop_net_repr.tune_hparams=True` and `cnf_repr.tune_hparams=True`.

#### Stage 2.
The bounds on the representation-induced confounding bias can be then estimated with the methods `calculate_gammas` and `get_bounds` from src/models/msm.py.

### Datasets
Before running semi-synthetic experiments, place datasets in the corresponding folders:
- [IHDP100 dataset](https://www.fredjo.com/): ihdp_npci_1-100.test.npz and ihdp_npci_1-100.train.npz to `data/ihdp100/`


One needs to specify a dataset / dataset generator (and some additional parameters, e.g. train size for the synthetic data `dataset.n_samples_train=1000`):
- Synthetic data (adapted from https://arxiv.org/abs/1810.02894): `+dataset=synthetic`
- [IHDP](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2010.08162) dataset: `+dataset=ihdp100` 
- [HC-MNIST](https://github.com/anndvision/quince/blob/main/quince/library/datasets/hcmnist.py) dataset: `+dataset=hcmnist`

### Examples
Example of running TARNet without tuning based on synthetic data with n_train = 1000:
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. python3 runnables/train.py -m +dataset=synthetic +repr_net=tarnet +repr_net/synthetic_hparams/tarnet/n1000='0.5' exp.logging=True exp.device=cuda exp.seed=10
```

Example of all-stages tuning of CFR based on the 0-th subset of IHDP100 dataset with Wasserstein metric and $\alpha = 1.0$:
```console
CUDA_VISIBLE_DEVICES=<devices> PYTHONPATH=. -m +dataset=ihdp100 +repr_net=cfrnet exp.logging=True exp.device=cuda dataset.dataset_ix=0 repr_net.ipm=wass repr_net.alpha=1.0 repr_net.tune_hparams=True prop_net_cov.tune_hparams=True prop_net_repr.tune_hparams=True cnf_repr.tune_hparams=True
```

-------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
