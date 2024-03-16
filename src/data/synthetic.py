import numpy as np
import pandas as pd

from src import ROOT_PATH


class SyntheticData:

    def __init__(self, n_samples_train, n_samples_test, **kwargs):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test

    def get_data(self):
        data_dicts = []
        for n_samples in [self.n_samples_train, self.n_samples_test]:
            u = np.random.normal(0.0, 1.0, n_samples)
            x = np.random.uniform(-2.0, 2.0, n_samples)
            t = np.random.binomial(1, 1.0 / (1.0 + np.exp(- (0.75 * x - u + 0.5))), n_samples)

            y_pot0 = self.get_mu(0, x, u) + np.random.normal(0.0, 1.0, n_samples)
            y_pot1 = self.get_mu(1, x, u) + np.random.normal(0.0, 1.0, n_samples)
            y = y_pot0 * (1 - t) + y_pot1 * t
            # y = (2 * t - 1) * x + (2 * t - 1) - 2 * np.sin(2 * (2 * t - 1) * x + u) - 2 * u * (1 + 0.5 * x) + y_eps
            data_dicts.append({
                'cov_f': np.stack([x, u], -1),
                'treat_f': t,
                'out_f': y,
                'out_pot0': y_pot0,
                'out_pot1': y_pot1,
                'mu0': self.get_mu(0, x, u),
                'mu1': self.get_mu(1, x, u),
            })
        return data_dicts

    def get_mu(self, treat, x, u):
        if treat == 0:
            return - 1 * x - 2 * np.sin(2 * - 1 * x + u) - 2 * u * (1 + 0.5 * x)
        else:
            return 1 * x + 1 - 2 * np.sin(2 * 1 * x + u) - 2 * u * (1 + 0.5 * x)

    def get_gt_cate(self, x, u):
        return self.get_mu(1, x, u) - self.get_mu(0, x, u)
