import numpy as np
from pyro.nn import DenseNN
import logging
import torch
import numpy as np
import pyro.distributions as dist
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger
from src.models.base_net import BaseNet
import pyro.distributions.transforms as T
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class MSM(BaseNet):
    val_metric = 'val_neg_log_prob'
    name = 'cnf_repr'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        self.network_type = args.repr_net.network_type  # Same as the representation network

        super(MSM, self).__init__(args, mlflow_logger)

        self.count_bins = args.cnf_repr.count_bins
        self.noise_std_X = args.cnf_repr.noise_std_X
        self.noise_std_Y = args.cnf_repr.noise_std_Y
        self.dim_repr = args.repr_net.dim_repr
        self.dim_hid1 = args[self.name].dim_hid1 = int(args[self.name].dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)
        self.bound = 5.0
        self.cond_base_dist = dist.Normal(torch.zeros((1, )), torch.ones((1, )))
        self.wd = 0.0

        if self.network_type == 'snet':
            self.cond_spline_nn = DenseNN(self.dim_repr + 1, [self.dim_hid1],
                                          param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)]).float()
            self.cond_spline_transform = T.ConditionalSpline(self.cond_spline_nn, 1, order='quadratic',
                                                             count_bins=self.count_bins,
                                                             bound=self.bound).to(self.device)
            self.cond_flow_dist = dist.ConditionalTransformedDistribution(self.cond_base_dist, [self.cond_spline_transform])

        elif self.network_type == 'tnet':
            self.cond_spline_nn = torch.nn.ModuleList([DenseNN(self.dim_repr, [self.dim_hid1], param_dims=[self.count_bins, self.count_bins, (self.count_bins - 1)]).float() for _ in self.treat_options])
            self.cond_spline_transform = torch.nn.ModuleList([T.ConditionalSpline(cond_spline_nn, 1, order='quadratic',
                                                             count_bins=self.count_bins,
                                                             bound=self.bound).to(self.device) for cond_spline_nn in self.cond_spline_nn])
            self.cond_flow_dist = [dist.ConditionalTransformedDistribution(self.cond_base_dist, [cond_spline_transform]) for (cond_spline_transform) in self.cond_spline_transform]
        else:
            raise NotImplementedError()

        self.out_scaler = StandardScaler()
        self.to(self.device)

    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        # Torch tensors
        _, treat_f, out_f = self.prepare_tensors(None, data_dict['treat_f'], out_f)
        repr_f = torch.tensor(data_dict['repr_f']).float()
        mu_pred0_repr, mu_pred1_repr = torch.tensor(data_dict['mu_pred0_repr']).float(), torch.tensor(data_dict['mu_pred1_repr']).float()

        return repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        out_f = self.out_scaler.transform(data_dict['out_f'].reshape(-1, 1))
        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        _, treat_f, out_f = self.prepare_tensors(None, data_dict['treat_f'], out_f)

        repr_f = torch.tensor(data_dict['repr_f']).float()
        out_pot0 = torch.tensor(out_pot0).reshape(-1, 1).float()
        out_pot1 = torch.tensor(out_pot1).reshape(-1, 1).float()
        mu0 = torch.tensor(mu0).reshape(-1, 1).float() if self.oracle_available else None
        mu1 = torch.tensor(mu1).reshape(-1, 1).float() if self.oracle_available else None

        mu_pred0_repr, mu_pred1_repr = torch.tensor(data_dict['mu_pred0_repr']).float(), torch.tensor(data_dict['mu_pred1_repr']).float()
        return repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr, out_pot0, out_pot1, mu0, mu1

    def get_train_dataloader(self, repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr):
        training_data = TensorDataset(repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader

    def fit(self, train_data_dict: dict, log: bool):
        repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr)
        optimizer = self.get_optimizer()

        # Logging
        # self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr = next(iter(train_dataloader))
            optimizer.zero_grad()

            noised_repr_f = repr_f + self.noise_std_X * torch.randn_like(repr_f)
            noised_out_f = out_f + self.noise_std_Y * torch.randn_like(out_f)

            loss, log_dict = self.forward_train(noised_repr_f, treat_f, noised_out_f, mu_pred0_repr, mu_pred1_repr, prefix='train')
            loss.backward()

            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def forward_train(self, repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr, prefix='train'):
        # Log-prob loss
        if self.network_type == 'tnet':
            log_prob = torch.stack([(treat_f == t) * self.cond_flow_dist[int(t)].condition(repr_f).log_prob(out_f) for t in self.treat_options], dim=0).sum(0)
        elif self.network_type == 'snet':
            context_f = torch.cat([repr_f, treat_f], dim=1)
            log_prob = self.cond_flow_dist.condition(context_f).log_prob(out_f)
        else:
            raise NotImplementedError()
        log_prob = log_prob.mean()
        results = {f'{prefix}_neg_log_prob': - log_prob.item()}
        return - log_prob, results

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr, _, _, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            _, results = self.forward_train(repr_f, treat_f, out_f, mu_pred0_repr, mu_pred1_repr, prefix=prefix)

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str, n_samples=1000):
        repr_f, _, _, _, _, out_pot0, out_pot1, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        mu_pred = [torch.zeros((repr_f.shape[0], 1)), torch.zeros((repr_f.shape[0], 1))]
        batch_size = 1000

        self.eval()
        for i in range(repr_f.shape[0] // batch_size + 1):
            repr_f_batch = repr_f[i * batch_size: (i + 1) * batch_size]
            with torch.no_grad():
                for t in self.treat_options:
                    if self.network_type == 'tnet':
                        out_sample = self.cond_flow_dist[int(t)].condition(repr_f_batch).sample((n_samples, repr_f_batch.shape[0]))
                    elif self.network_type == 'snet':
                        context = torch.cat([repr_f_batch, t * torch.ones((repr_f_batch.shape[0], 1))], dim=1)
                        out_sample = self.cond_flow_dist.condition(context).sample((n_samples, repr_f_batch.shape[0]))
                    else:
                        raise NotImplementedError()
                    mu_pred[(int(t))][i * batch_size: (i + 1) * batch_size] = out_sample.mean(0)

        rpehe = (((mu_pred[1] - mu_pred[0]) - (out_pot1 - out_pot0)) ** 2).mean().sqrt()

        results = {f'{prefix}_cnf_rpehe': rpehe.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results

    def evaluate_policy(self, data_dict: dict, delta: float, log: bool, prefix: str):
        repr_f, _, _, _, _, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        lb = data_dict[f'mu_pred1_PSM_bounds_{delta}'][0] - data_dict[f'mu_pred0_PSM_bounds_{delta}'][1]
        ub = data_dict[f'mu_pred1_PSM_bounds_{delta}'][1] - data_dict[f'mu_pred0_PSM_bounds_{delta}'][0]

        deferral_mask = ((lb <= 0.0) & (ub >= 0.0))
        deferral_rate = deferral_mask.mean()

        # Default policy is 0
        lb = torch.tensor(lb).reshape(-1, 1)
        ub = torch.tensor(ub).reshape(-1, 1)
        # policy_val_default = (out_pot1 * (lb > 0.0) + out_pot0 * (lb <= 0.0)).mean()
        # policy_val_deferral = (out_pot1[~deferral_mask] * (lb[~deferral_mask] > 0.0) + out_pot0[~deferral_mask] * (ub[~deferral_mask] < 0.0)).mean()

        cate_gt = mu1 - mu0
        error_rate = 1.0 - ((cate_gt[~deferral_mask] > 0.0) == ((lb[~deferral_mask] > 0.0))).float().mean()

        results = {
            # f'{prefix}_cnf_pol_val_default': policy_val_default.item(),
            # f'{prefix}_cnf_pol_val_deferral': policy_val_deferral.item(),
            f'{prefix}_cnf_error_rate_{delta}': error_rate.item(),
            f'{prefix}_cnf_deferral_rate_{delta}': deferral_rate
        }
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results


    @staticmethod
    def calculate_gammas(data_dict, reference_data_dict=None, clip_prop=0.0, set_nan_to=1000.0, delta=0.05):
        prop_pred_cov, prop_pred_repr = data_dict['prop_pred_cov'], data_dict['prop_pred_repr']
        mask = (prop_pred_cov > clip_prop) & (prop_pred_repr > clip_prop) & (1.0 - prop_pred_cov > clip_prop) & (
                1.0 - prop_pred_repr > clip_prop)
        if mask.sum() > 0:
            logger.warning('Propensity scores reaching 0 or 1!')
        ratio = (1.0 - prop_pred_repr) * prop_pred_cov / prop_pred_repr / (1.0 - prop_pred_cov)
        gammas = np.maximum(ratio, (1 / ratio))
        gammas[np.isnan(gammas) | np.isinf(gammas)] = set_nan_to

        if reference_data_dict is None:
            return gammas.squeeze()
        else:
            reference_gammas = MSM.calculate_gammas(reference_data_dict, reference_data_dict=None, clip_prop=clip_prop, set_nan_to=set_nan_to)
            repr_dist = cdist(data_dict['repr_f'], reference_data_dict['repr_f'])
            repr_dist_mask = np.where(repr_dist < delta, 1.0, 0.0)
            max_agg_gammas = (reference_gammas[None, :] * repr_dist_mask).max(1)
            gammas = np.maximum(gammas.squeeze(), max_agg_gammas)
            return gammas



    def get_bounds(self, data_dict, delta, n_samples=1000):
        repr_f, _, _, _, _, _, _, _, _ = self.prepare_eval_data(data_dict)
        prop_pred_repr = torch.tensor(data_dict['prop_pred_repr']).float()
        gammas = torch.tensor(data_dict[f'gammas_{delta}']).float()
        bounds = [[np.zeros(repr_f.shape[0]), np.zeros(repr_f.shape[0])], [np.zeros(repr_f.shape[0]), np.zeros(repr_f.shape[0])]]
        batch_size = 1000

        self.eval()
        for i in range(repr_f.shape[0] // batch_size + 1):
            repr_f_batch = repr_f[i * batch_size: (i + 1) * batch_size]
            gammas_batch = gammas[i * batch_size: (i + 1) * batch_size]
            prop_pred_repr_batch = prop_pred_repr[i * batch_size: (i + 1) * batch_size]

            with torch.no_grad():
                for t in self.treat_options:
                    q_t = prop_pred_repr_batch.squeeze() if t == 1.0 else 1.0 - prop_pred_repr_batch.squeeze()
                    s_minus, s_plus = 1.0 / ((1.0 - gammas_batch) * q_t + gammas_batch), 1.0 / ((1.0 - 1.0 / gammas_batch) * q_t + 1.0 / gammas_batch)
                    c_minus, c_plus = 1.0 / (1.0 + gammas_batch), gammas_batch / (1.0 + gammas_batch)

                    q_index_minus, q_index_plus = torch.floor(c_minus * n_samples).long(), torch.floor(c_plus * n_samples).long()
                    q_index_minus, q_index_plus = torch.clamp(q_index_minus, 0, n_samples - 1), torch.clamp(q_index_plus, 0, n_samples - 1)

                    if self.network_type == 'tnet':
                        out_sample = self.cond_flow_dist[int(t)].condition(repr_f_batch).sample((n_samples, repr_f_batch.shape[0])).squeeze()
                    elif self.network_type == 'snet':
                        context = torch.cat([repr_f_batch, t * torch.ones((repr_f_batch.shape[0], 1))], dim=1)
                        out_sample = self.cond_flow_dist.condition(context).sample((n_samples, repr_f_batch.shape[0])).squeeze()
                    else:
                        raise NotImplementedError()

                    out_sample = out_sample.sort(0)[0]
                    out_sample_cum = torch.cumsum(out_sample, dim=0)

                    out_sample_cum_minus = out_sample_cum.gather(dim=0, index=q_index_minus.unsqueeze(0)).squeeze()
                    out_sample_cum_plus = out_sample_cum.gather(dim=0, index=q_index_plus.unsqueeze(0)).squeeze()
                    out_sample_cum_all = out_sample_cum[-1, :]

                    mu_minus = (out_sample_cum_minus / (n_samples * s_minus)) + \
                                ((out_sample_cum_all - out_sample_cum_minus) / (n_samples * s_plus))
                    mu_plus = (out_sample_cum_plus / (n_samples * s_plus)) + \
                               ((out_sample_cum_all - out_sample_cum_plus) / (n_samples * s_minus))

                    large_gamma_mask = mu_minus > mu_plus
                    mu_minus[large_gamma_mask], mu_plus[large_gamma_mask] = out_sample[0, large_gamma_mask], out_sample[-1, large_gamma_mask]

                    bounds[int(t)][0][i * batch_size: (i + 1) * batch_size] = mu_minus.cpu().numpy()
                    bounds[int(t)][1][i * batch_size: (i + 1) * batch_size] = mu_plus.cpu().numpy()

        return bounds[0], bounds[1]

    def get_coverage(self, data_dict):

        pred_cover0 = ((data_dict['mu_pred0_repr'].squeeze() > data_dict['mu_pred0_PSM_bounds'][0]) & (data_dict['mu_pred0_repr'].squeeze() < data_dict['mu_pred0_PSM_bounds'][1])).mean()
        pred_cover1 = ((data_dict['mu_pred1_repr'].squeeze() > data_dict['mu_pred1_PSM_bounds'][0]) & (data_dict['mu_pred1_repr'].squeeze() < data_dict['mu_pred1_PSM_bounds'][1])).mean()
        pred_cover = 0.5 * (pred_cover0 + pred_cover1)

        mu0, mu1 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)), self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1))

        gt_cover0 = ((mu0.squeeze() > data_dict['mu_pred0_PSM_bounds'][0]) & (mu0.squeeze() < data_dict['mu_pred0_PSM_bounds'][1])).mean()
        gt_cover1 = ((mu1.squeeze() > data_dict['mu_pred1_PSM_bounds'][0]) & (mu1.squeeze() < data_dict['mu_pred1_PSM_bounds'][1])).mean()
        gt_cover = 0.5 * (gt_cover0 + gt_cover1)

        return pred_cover, gt_cover

