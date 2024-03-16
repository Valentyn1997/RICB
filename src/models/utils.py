import numpy as np
import ot
from torch import nn

import torch
from copy import deepcopy
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from ray import tune


def fit_eval_kfold(args: dict, orig_hparams: DictConfig, model_cls, train_data_dict: dict, val_data_dict: dict, name: str,
                   kind: str = None, subnet_name: str = None, **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param orig_hparams: DictConfig of original hyperparameters
    :param model_cls: class of model
    :param kwargs: Other args
    """
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params[name], args)
    model_cls.set_subnet_hparams(new_params[subnet_name], args) if subnet_name is not None else None
    new_params.exp.device = 'cpu'  # Tuning only with cpu

    if val_data_dict is None:  # KFold hparam tuning
        kf = KFold(n_splits=5, random_state=orig_hparams.exp.seed, shuffle=True)
        val_metrics = []
        for train_index, val_index in kf.split(train_data_dict['cov_f']):
            ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, train_index), \
                                              subset_by_indices(train_data_dict, val_index)

            model = model_cls(new_params, kind=kind, **kwargs)
            model.fit(train_data_dict=ttrain_data_dict, log=False)
            log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
            val_metrics.append(log_dict[model.val_metric])
        tune.report(val_metric=np.mean(val_metrics))

    else:  # predefined hold-out hparam tuning
        model = model_cls(new_params, kind=kind, **kwargs)
        model.fit(train_data_dict=train_data_dict, log=False)
        log_dict = model.evaluate(data_dict=val_data_dict, log=False, prefix='val')
        tune.report(val_metric=log_dict[model.val_metric])


def subset_by_indices(data_dict: dict, indices: list):
    subset_data_dict = {}
    for (k, v) in data_dict.items():
        subset_data_dict[k] = np.copy(data_dict[k][indices])
    return subset_data_dict


def wass_dist(repr_f, treat_f, weights=None):
    repr0, repr1 = repr_f[treat_f.squeeze() == 0.0, :], repr_f[treat_f.squeeze() == 1.0]
    min_size = min(repr0.shape[0], repr1.shape[0])
    repr0, repr1 = repr0[:min_size, :], repr1[:min_size, :]
    M = ot.dist(repr0, repr1)

    if weights is None:
        w0, w1 = torch.ones(min_size), torch.ones(min_size)
    else:
        w0, w1 = weights[treat_f.squeeze() == 0.0, :], weights[treat_f.squeeze() == 1.0]
        w0, w1 = w0[:min_size, 0] + 1e-9, w1[:min_size, 0] + 1e-9

    wass_dist = ot.emd2(w0 / (w0.sum()), w1 / (w1.sum()), M)
    return wass_dist


def mmd_dist(repr_f, treat_f, weights=None):

    class RBF(nn.Module):

        def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
            super().__init__()
            self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
            self.bandwidth = bandwidth

        def get_bandwidth(self, L2_distances):
            if self.bandwidth is None:
                n_samples = L2_distances.shape[0]
                return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

            return self.bandwidth

        def forward(self, X):
            L2_distances = torch.cdist(X, X) ** 2
            return torch.exp(
                -L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(
                dim=0)

    class MMDLoss(nn.Module):

        def __init__(self, kernel=RBF()):
            super().__init__()
            self.kernel = kernel

        def forward(self, X, Y):
            K = self.kernel(torch.vstack([X, Y]))

            X_size = X.shape[0]
            XX = K[:X_size, :X_size].mean()
            XY = K[:X_size, X_size:].mean()
            YY = K[X_size:, X_size:].mean()
            return XX - 2 * XY + YY

    repr0, repr1 = repr_f[treat_f.squeeze() == 0.0, :], repr_f[treat_f.squeeze() == 1.0]
    min_size = min(repr0.shape[0], repr1.shape[0])
    repr0, repr1 = repr0[:min_size, :], repr1[:min_size, :]
    mmd_dist = MMDLoss()(repr0, repr1)
    return mmd_dist
