import numpy as np
import pandas as pd

from src import ROOT_PATH


class IHDP100:

    def __init__(self, **kwargs):
        self.train_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.train.npz"
        self.test_data_path = f"{ROOT_PATH}/data/ihdp100/ihdp_npci_1-100.test.npz"

    def get_data(self):
        train_data = np.load(self.train_data_path, 'r')
        test_data = np.load(self.test_data_path, 'r')

        datasets = []

        for i in range(train_data['x'].shape[-1]):

            data_dicts = []

            for data in [train_data, test_data]:

                data_dicts.append({
                    'cov_f': data['x'][:, :, i],
                    'treat_f': data['t'][:, i],
                    'out_f': data['yf'][:, i],
                    'out_pot0': np.where(1.0 - data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'out_pot1': np.where(data['t'][:, i], data['yf'][:, i], data['ycf'][:, i]),
                    'mu0': data['mu0'][:, i],
                    'mu1': data['mu1'][:, i],
                })

            datasets.append(data_dicts)

        return datasets
