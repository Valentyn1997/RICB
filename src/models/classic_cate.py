import numpy as np
import ray
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune
import logging
from pytorch_lightning.loggers import MLFlowLogger

from econml.grf import CausalForest
# from xbcausalforest import XBCF
from sklearn.neighbors import KNeighborsRegressor

from src.models.utils import fit_eval_kfold

logger = logging.getLogger(__name__)


class CATEEstimator:

    val_metric = None
    name = 'cate_estimator'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):

        self.dim_cov = args.dataset.dim_cov
        self.cov_scaler = StandardScaler()
        self.out_scaler = StandardScaler()
        self.treat_options = [0.0, 1.0]
        self.oracle_available = args.dataset.oracle_available

        # CATE estimator
        self.hparams = args
        self.estimator = None

        # MlFlow Logger
        self.mlflow_logger = mlflow_logger

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            assert k in model_args.keys()
            model_args[k] = new_model_args[k]

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        return cov_f, data_dict['treat_f'], out_f

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.transform(data_dict['out_f'].reshape(-1, 1))

        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        return cov_f, data_dict['treat_f'], out_f, out_pot0, out_pot1, mu0, mu1

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        raise NotImplementedError()

    def predict_cate(self, cov_f: np.array):
        raise NotImplementedError()

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str):
        cov_f, _, _, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        rpehe = np.sqrt(((self.predict_cate(cov_f) - (out_pot1 - out_pot0)) ** 2).mean())
        rpehe_oracle = np.sqrt((((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean()) if self.oracle_available else None

        results = {f'{prefix}_rpehe': rpehe.item(), f'{prefix}_rpehe_oracle': rpehe_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results)
        return results

    def evaluate_policy(self, data_dict: dict, log: bool, prefix: str):
        cov_f, _, _, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        # policy_val_pred = (out_pot1 * (cate_pred > 0.0) + out_pot0 * (cate_pred <= 0.0)).mean()

        cate_gt = mu1 - mu0
        # policy_val_gt = (out_pot1 * (cate_gt > 0.0) + out_pot0 * (cate_gt <= 0.0)).mean()

        error_rate = 1.0 - ((self.predict_cate(cov_f) > 0.0) == (cate_gt > 0.0)).mean()

        results = {
            # f'{prefix}_pol_val': policy_val_pred.item(),
            # f'{prefix}_pol_val_gt': policy_val_gt.item(),
            f'{prefix}_error_rate': error_rate
        }
        if log:
            self.mlflow_logger.log_metrics(results)
        return results

    def finetune(self, train_data_dict: dict, resources_per_trial: dict, val_data_dict: dict = None):
        """
        Hyperparameter tuning with ray[tune]
        """

        logger.info(f"Running hyperparameters selection with {self.hparams[self.name]['tune_range']} trials")
        logger.info(f'Using {self.val_metric} for hyperparameters selection')
        ray.init(num_gpus=0, num_cpus=4)

        hparams_grid = {k: getattr(tune, self.hparams[self.name]['tune_type'])(list(v))
                        for k, v in self.hparams[self.name]['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(fit_eval_kfold,
                                                 model_cls=self.__class__,
                                                 train_data_dict=deepcopy(train_data_dict),
                                                 val_data_dict=deepcopy(val_data_dict),
                                                 name=self.name,
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
        self.set_hparams(self.hparams[self.name], analysis.best_config)
        self.__init__(self.hparams, mlflow_logger=self.mlflow_logger)
        return self


class CForest(CATEEstimator):

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(CForest, self).__init__(args, mlflow_logger)
        self.estimator = CausalForest(n_estimators=args.cate_estimator.n_estimators,
                                      min_samples_leaf=args.cate_estimator.min_samples_leaf,
                                      min_balancedness_tol=args.cate_estimator.min_balancedness_tol)

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        self.estimator.fit(cov_f, treat_f, out_f)

    def predict_cate(self, cov_f: np.array):
        return self.estimator.predict(cov_f)


# class BART(CATEEstimator):
#     def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
#         super(BART, self).__init__(args, mlflow_logger)
#
#         self.estimator = XBCF(num_sweeps=args.cate_estimator.num_sweeps,
#                               num_trees_pr=args.cate_estimator.num_trees_pr,
#                               num_trees_trt=args.cate_estimator.num_trees_trt,
#                               Nmin=args.cate_estimator.min_samples_leaf,
#                               # alpha_pr=0.95,  # shrinkage (splitting probability)
#                               # beta_pr=2,  # shrinkage (tree depth)
#                               # alpha_trt=0.95,  # shrinkage for treatment part
#                               # beta_trt=2,
#                               # max_depth=500,
#                               max_depth=args.cate_estimator.max_depth,
#                               parallel=True,
#                               p_categorical_pr=0,
#                               p_categorical_trt=0,
#                               standardize_target=False)
#
#     def fit(self, train_data_dict: dict, log: bool):
#         cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
#
#         self.estimator.fit(x_t=cov_f.astype('float32'),
#                            x=cov_f.astype('float32'),
#                            y=out_f.reshape(-1).astype('float32'),
#                            z=treat_f.reshape(-1).astype('int32'))
#
#     def predict_cate(self, cov_f: np.array):
#         return self.estimator.predict(cov_f.astype('float32')).reshape(-1, 1)


class KNN(CATEEstimator):
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(KNN, self).__init__(args, mlflow_logger)
        self.estimator_0 = KNeighborsRegressor(n_neighbors=args.cate_estimator.n_neighbors)
        self.estimator_1 = KNeighborsRegressor(n_neighbors=args.cate_estimator.n_neighbors)

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)

        self.estimator_0.fit(cov_f[treat_f == 0.0], out_f[treat_f == 0.0])
        self.estimator_1.fit(cov_f[treat_f == 1.0], out_f[treat_f == 1.0])

    def predict_cate(self, cov_f: np.array):
        return self.estimator_1.predict(cov_f) - self.estimator_0.predict(cov_f)
