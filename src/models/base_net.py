import numpy as np
import torch
import ray
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune
import logging
from pytorch_lightning.loggers import MLFlowLogger

from src.models.utils import fit_eval_kfold

logger = logging.getLogger(__name__)


class BaseNet(torch.nn.Module):

    val_metric = None
    name = None
    subnet_name = None

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(BaseNet, self).__init__()

        self.dim_cov = args.dataset.dim_cov
        self.cov_scaler = StandardScaler()
        self.device = args.exp.device
        self.treat_options = [0.0, 1.0]
        self.oracle_available = args.dataset.oracle_available

        # Model hyparams
        self.hparams = args
        self.num_epochs = args[self.name].num_epochs
        self.batch_size = args[self.name].batch_size
        self.lr = args[self.name].lr

        # MlFlow Logger
        self.mlflow_logger = mlflow_logger

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            assert k in model_args.keys()
            model_args[k] = new_model_args[k]

    @staticmethod
    def set_subnet_hparams(model_args: DictConfig, new_model_args: dict):
        pass

    def prepare_tensors(self, cov=None, treat=None, out=None):
        cov = torch.tensor(cov).reshape(-1, self.dim_cov).float() if cov is not None else None
        treat = torch.tensor(treat).reshape(-1, 1).float() if treat is not None else None
        out = torch.tensor(out).reshape(-1, 1).float() if out is not None else None
        return cov, treat, out

    def prepare_train_data(self, data_dict: dict):
        raise NotImplementedError()

    def prepare_eval_data(self, data_dict: dict):
        raise NotImplementedError()

    def fit(self, train_data_dict: dict, log: bool):
        raise NotImplementedError()

    def forward_train(self, repr_f, treat_f, out_f, cov_f, prefix='train'):
        raise NotImplementedError()

    def forward_eval(self, repr_f):
        raise NotImplementedError()

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        raise NotImplementedError()

    def get_representations(self, data_dict):
        raise NotImplementedError()

    def get_outcomes(self, data_dict):
        raise NotImplementedError()

    def get_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def finetune(self, train_data_dict: dict, resources_per_trial: dict, val_data_dict: dict = None):
        """
        Hyperparameter tuning with ray[tune]
        """

        logger.info(f"Running hyperparameters selection with {self.hparams[self.name]['tune_range']} trials")
        logger.info(f'Using {self.val_metric} for hyperparameters selection')
        ray.init(num_gpus=0, num_cpus=4)

        hparams_grid = {k: getattr(tune, self.hparams[self.name]['tune_type'])(list(v))
                        for k, v in self.hparams[self.name]['hparams_grid'].items()}
        torch.set_default_device('cpu')
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 val_data_dict=deepcopy(val_data_dict),
                                                 name=self.name, subnet_name=self.subnet_name,
                                                 kind=self.kind if hasattr(self, 'kind') else None,
                                                 orig_hparams=self.hparams),
                            resources_per_trial=resources_per_trial,
                            raise_on_failed_trial=False,
                            metric='val_metric',
                            mode="min",
                            config=hparams_grid,
                            num_samples=self.hparams[self.name]['tune_range'],
                            name=f"{self.__class__.__name__}",
                            max_failures=1,
                            )
        ray.shutdown()

        logger.info(f"Best hyperparameters found for {self.name}: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        torch.set_default_device(self.device)
        self.set_hparams(self.hparams[self.name], analysis.best_config)
        self.set_subnet_hparams(self.hparams[self.subnet_name], analysis.best_config) if self.subnet_name is not None else None
        self.__init__(self.hparams, mlflow_logger=self.mlflow_logger, kind=self.kind if hasattr(self, 'kind') else None)
        return self
