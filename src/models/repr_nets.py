from pyro.nn import DenseNN
import logging
import torch
import numpy as np
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning.loggers import MLFlowLogger

from src.models.base_net import BaseNet
from src.models.prop_nets import PropNet
from src.models.utils import wass_dist, mmd_dist

logger = logging.getLogger(__name__)


class BaseRepresentationNet(BaseNet):

    val_metric = 'val_rmse'
    name = 'repr_net'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(BaseRepresentationNet, self).__init__(args, mlflow_logger)

        self.dim_hid1 = args.repr_net.dim_hid1 = int(args.repr_net.dim_hid1_multiplier * args.dataset.extra_hid_multiplier * self.dim_cov)
        self.dim_repr = args.repr_net.dim_repr = int(args.repr_net.dim_repr_multiplier * self.dim_cov)
        self.dim_hid2 = args.repr_net.dim_hid2 = int(args.repr_net.dim_hid2_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)
        self.wd = args.repr_net.wd

        self.out_scaler = StandardScaler()

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        # Torch tensors
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, data_dict['treat_f'], out_f)

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        return cov_f, treat_f, out_f

    def prepare_eval_data(self, data_dict: dict):
        # Scaling eval data
        cov_f = self.cov_scaler.transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.transform(data_dict['out_f'].reshape(-1, 1))
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, data_dict['treat_f'], out_f)

        out_pot0 = self.out_scaler.transform(data_dict['out_pot0'].reshape(-1, 1))
        out_pot1 = self.out_scaler.transform(data_dict['out_pot1'].reshape(-1, 1))
        mu0 = self.out_scaler.transform(data_dict['mu0'].reshape(-1, 1)) if self.oracle_available else None
        mu1 = self.out_scaler.transform(data_dict['mu1'].reshape(-1, 1)) if self.oracle_available else None

        out_pot0 = torch.tensor(out_pot0).reshape(-1, 1).float()
        out_pot1 = torch.tensor(out_pot1).reshape(-1, 1).float()
        mu0 = torch.tensor(mu0).reshape(-1, 1).float() if self.oracle_available else None
        mu1 = torch.tensor(mu1).reshape(-1, 1).float() if self.oracle_available else None

        return cov_f, treat_f, out_f, out_pot0, out_pot1, mu0, mu1

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)
        optimizer = self.get_optimizer()

        # Logging
        # self.mlflow_logger.log_hyperparams(self.hparams) if log else None

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            optimizer.zero_grad()

            repr_f = self.repr_nn(cov_f)
            loss, log_dict = self.forward_train(repr_f, treat_f, out_f, cov_f, prefix='train')
            loss.backward()

            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        cov_f, treat_f, out_f, _, _, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            _, results = self.forward_train(repr_f, treat_f, out_f, cov_f, prefix=prefix)
            results[f'{prefix}_rmse'] = np.sqrt(results[f'{prefix}_mse'])

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results

    def evaluate_pehe(self, data_dict: dict, log: bool, prefix: str):
        cov_f, _, _, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            mu_pred0, mu_pred1 = self.forward_eval(repr_f)

            rpehe = (((mu_pred1 - mu_pred0) - (out_pot1 - out_pot0)) ** 2).mean().sqrt()
            rpehe_oracle = (((mu1 - mu0) - (out_pot1 - out_pot0)) ** 2).mean().sqrt() if self.oracle_available else None

        results = {f'{prefix}_rpehe': rpehe.item(), f'{prefix}_rpehe_oracle': rpehe_oracle.item()}
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results

    def evaluate_policy(self, data_dict: dict, log: bool, prefix: str):
        cov_f, _, _, out_pot0, out_pot1, mu0, mu1 = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            mu_pred0, mu_pred1 = self.forward_eval(repr_f)

            cate_pred = mu_pred1 - mu_pred0
            # policy_val_pred = (out_pot1 * (cate_pred > 0.0) + out_pot0 * (cate_pred <= 0.0)).mean()

            cate_gt = mu1 - mu0
            # policy_val_gt = (out_pot1 * (cate_gt > 0.0) + out_pot0 * (cate_gt <= 0.0)).mean()

            error_rate = 1.0 - ((cate_pred > 0.0) == (cate_gt > 0.0)).float().mean()

        results = {
            # f'{prefix}_pol_val': policy_val_pred.item(),
            # f'{prefix}_pol_val_gt': policy_val_gt.item(),
            f'{prefix}_error_rate': error_rate.item()
        }
        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)
        return results

    def get_representations(self, data_dict):
        cov_f, _, _, _, _, _, _ = self.prepare_eval_data(data_dict)
        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
        return repr_f.cpu().numpy()

    def get_outcomes(self, data_dict):
        cov_f, _, _, _, _, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            mu_pred0, mu_pred1 = self.forward_eval(repr_f)
        return mu_pred0.cpu().numpy(), mu_pred1.cpu().numpy()

    def get_train_dataloader(self, cov_f, treat_f, out_f):
        training_data = TensorDataset(cov_f, treat_f, out_f)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader


class TARNet(BaseRepresentationNet):
    """Repr + TNet"""
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        # Model hyparams & Train params

        super(TARNet, self).__init__(args, mlflow_logger)
        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid1], param_dims=[self.dim_repr], nonlinearity=torch.nn.ELU()).float()
        self.mu_nn = torch.nn.ModuleList([DenseNN(self.dim_repr, [self.dim_hid2], param_dims=[1], nonlinearity=torch.nn.ELU()).float() for _ in self.treat_options])

        self.to(self.device)

    def forward_train(self, repr_f, treat_f, out_f, cov_f, prefix='train'):
        mu_pred = torch.stack([(treat_f == t) * self.mu_nn[int(t)](repr_f) for t in self.treat_options], dim=0).sum(0)
        mse = ((mu_pred - out_f) ** 2)
        mse = mse.mean()

        results = {f'{prefix}_mse': mse.item()}
        return mse, results

    def forward_eval(self, repr_f):
        mu_pred_0, mu_pred_1 = self.mu_nn[0](repr_f), self.mu_nn[1](repr_f)
        return mu_pred_0, mu_pred_1


class BNNet(BaseRepresentationNet):
    """Repr + SNet + IMP(Repr)"""
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(BNNet, self).__init__(args, mlflow_logger)

        self.repr_nn = DenseNN(self.dim_cov, [self.dim_hid1], param_dims=[self.dim_repr], nonlinearity=torch.nn.ELU()).float()
        self.mu_nn = DenseNN(self.dim_repr + 1, [self.dim_hid2], param_dims=[1], nonlinearity=torch.nn.ELU()).float()

        self.alpha = args.repr_net.alpha
        self.ipm = args.repr_net.ipm
        assert self.ipm in ['wass', 'mmd']

        self.to(self.device)

    def forward_train(self, repr_f, treat_f, out_f, cov_f, prefix='train'):
        context_f = torch.cat([repr_f, treat_f], dim=1)

        mu_pred = self.mu_nn(context_f)
        mse = ((mu_pred - out_f) ** 2)
        mse = mse.mean()

        if self.ipm == 'wass':
            ipm = wass_dist(repr_f, treat_f)
        else:
            ipm = mmd_dist(repr_f, treat_f)

        results = {f'{prefix}_mse': mse.item(), f'{prefix}_imp': ipm.item()}
        return mse + self.alpha * ipm, results

    def forward_eval(self, repr_f):
        context_0 = torch.cat([repr_f, torch.zeros((repr_f.shape[0], 1))], dim=1)
        context_1 = torch.cat([repr_f, torch.ones((repr_f.shape[0], 1))], dim=1)

        mu_pred_0, mu_pred_1 = self.mu_nn(context_0), self.mu_nn(context_1)
        return mu_pred_0, mu_pred_1


class InvTARNet(TARNet):
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(InvTARNet, self).__init__(args, mlflow_logger)

        self.alpha_inv = args.repr_net.alpha_inv

        self.inv_repr_nn = DenseNN(self.dim_repr, [self.dim_hid1], param_dims=[self.dim_cov], nonlinearity=torch.nn.ELU()).float()
        self.to(self.device)

    def forward_train(self, repr_f, treat_f, out_f, cov_f, prefix='train'):
        mu_pred = torch.stack([(treat_f == t) * self.mu_nn[int(t)](repr_f) for t in self.treat_options], dim=0).sum(0)
        mse = ((mu_pred - out_f) ** 2)
        mse = mse.mean()

        cov_pred_f = self.inv_repr_nn(repr_f)
        mse_rec = ((cov_pred_f - cov_f) ** 2)
        mse_rec = mse_rec.mean()

        results = {f'{prefix}_mse': mse.item(), f'{prefix}_mse_rec': mse_rec.item()}
        return mse + self.alpha_inv * mse_rec, results


class CFRNet(TARNet):
    """Repr + TNet + IMP(Repr)"""
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(CFRNet, self).__init__(args, mlflow_logger)

        self.alpha = args.repr_net.alpha
        self.ipm = args.repr_net.ipm
        assert self.ipm in ['wass', 'mmd']

    def forward_train(self, repr_f, treat_f, out_f, cov_f, weights=None, prefix='train'):
        mu_pred = torch.stack([(treat_f == t) * self.mu_nn[int(t)](repr_f) for t in self.treat_options], dim=0).sum(0)
        mse = ((mu_pred - out_f) ** 2)
        if weights is None:
            mse = mse.mean()
        else:
            mse = (weights / weights.sum() * mse).sum()

        if self.ipm == 'wass':
            ipm = wass_dist(repr_f, treat_f, weights)
        else:
            ipm = mmd_dist(repr_f, treat_f, weights)

        results = {f'{prefix}_mse': mse.item(), f'{prefix}_imp': ipm.item()}
        return mse + self.alpha * ipm, results


class RCFRNet(CFRNet):
    """Repr + TNet + IMP(Repr) + Re-weighting(Repr)"""
    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(RCFRNet, self).__init__(args, mlflow_logger)

        self.weight_nn_lr = args.repr_net.weight_nn_lr
        self.weight_nn_wd = args.repr_net.weight_nn_wd

        self.dim_hid3 = args.repr_net.dim_hid3 = int(args.repr_net.dim_hid3_multiplier * args.dataset.extra_hid_multiplier * self.dim_repr)
        self.weight_nn = DenseNN(self.dim_repr, [self.dim_hid3], param_dims=[1], nonlinearity=torch.nn.ELU()).float()

        self.to(self.device)

    def get_optimizer(self):
        return [torch.optim.AdamW(list(self.repr_nn.parameters()) + list(self.mu_nn.parameters()), lr=self.lr, weight_decay=self.wd),
                torch.optim.AdamW(self.weight_nn.parameters(), lr=self.weight_nn_lr, weight_decay=self.weight_nn_wd)]

    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)
        cfr_optimizer, weight_optimizer = self.get_optimizer()

        weights = None

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            # Optimization step for cfrnet
            cfr_optimizer.zero_grad()

            repr_f = self.repr_nn(cov_f)

            if step > 0:
                weights = torch.nn.Softplus()(self.weight_nn(repr_f)).detach()

            loss, log_dict = self.forward_train(repr_f, treat_f, out_f, cov_f, weights, prefix='train')

            loss.backward()
            cfr_optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

            # Optimization step for weight_nn
            weight_optimizer.zero_grad()
            weights = torch.nn.Softplus()(self.weight_nn(repr_f.detach()))

            if self.ipm == 'wass':
                ipm = wass_dist(repr_f.detach(), treat_f, weights)
            else:
                ipm = mmd_dist(repr_f.detach(), treat_f, weights)

            ipm.backward()
            weight_optimizer.step()


class CFRISW(CFRNet):
    """Repr + TNet + IMP(Repr) + Re-weighting(Repr)"""

    val_metric = 'val_rmse_bce_repr'
    subnet_name = 'prop_net_repr'

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(CFRISW, self).__init__(args, mlflow_logger)

        self.prop_net_repr = PropNet(args, mlflow_logger, kind='repr')
        self.to(self.device)

    def fit(self, train_data_dict: dict, log: bool):
        self.prop_net_repr.prepare_train_data(train_data_dict)
        cov_f, treat_f, out_f = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f)
        cfr_optimizer, prop_net_repr_optimizer = self.get_optimizer()

        p1 = treat_f.mean()
        weights = None

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f = next(iter(train_dataloader))
            # Optimization step for cfrnet
            cfr_optimizer.zero_grad()

            repr_f = self.repr_nn(cov_f)

            if step > 0:
                with torch.no_grad():
                    p_f = (p1 * treat_f + (1 - p1) * (1 - treat_f))
                    pi1 = torch.sigmoid(self.prop_net_repr.prop_nn(repr_f))
                    pi_f = (pi1 * treat_f + (1 - pi1) * (1 - treat_f))
                    weights = 1 + p_f / (1 - p_f + 1e-9) * (1 - pi_f) / (pi_f + 1e-9)

            loss, log_dict = self.forward_train(repr_f, treat_f, out_f, cov_f, weights, prefix='train')

            loss.backward()
            cfr_optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)

            # Optimization step for prop_net_repr
            prop_net_repr_optimizer.zero_grad()

            prop_preds = self.prop_net_repr.prop_nn(repr_f.detach())
            loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()
            log_dict = {f'train_bce_repr': loss.item()}

            loss.backward()
            prop_net_repr_optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)


    def evaluate(self, data_dict: dict, log: bool, prefix: str):
        cov_f, treat_f, out_f, _, _, _, _ = self.prepare_eval_data(data_dict)

        self.eval()
        with torch.no_grad():
            repr_f = self.repr_nn(cov_f)
            _, results = self.forward_train(repr_f, treat_f, out_f, cov_f, prefix=prefix)
            results[f'{prefix}_rmse'] = np.sqrt(results[f'{prefix}_mse'])

            prop_preds = self.prop_net_repr.prop_nn(repr_f)
            loss = torch.binary_cross_entropy_with_logits(prop_preds, treat_f).mean()
            results[f'{prefix}_bce_repr'] = loss.item()

            results[f'{prefix}_rmse_bce_repr'] = results[f'{prefix}_rmse'] + results[f'{prefix}_bce_repr']

        if log:
            self.mlflow_logger.log_metrics(results, step=self.num_epochs)

        return results

    def get_optimizer(self):
        return [torch.optim.AdamW(list(self.repr_nn.parameters()) + list(self.mu_nn.parameters()), lr=self.lr, weight_decay=self.wd),
                self.prop_net_repr.get_optimizer()]

    @staticmethod
    def set_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            if CFRISW.subnet_name not in k:
                assert k in model_args.keys()
                model_args[k] = new_model_args[k]

    @staticmethod
    def set_subnet_hparams(model_args: DictConfig, new_model_args: dict):
        for k in new_model_args.keys():
            if CFRISW.subnet_name in k:
                k = k.split(CFRISW.subnet_name + '_')[-1]
                assert k in model_args.keys()
                model_args[k] = new_model_args[k]
        model_args['batch_size'] = new_model_args['batch_size']


class BWCFR(CFRNet):
    """Repr + TNet + IMP(Repr) + Re-weighting(Cov)"""

    def __init__(self, args: DictConfig = None, mlflow_logger: MLFlowLogger = None, **kwargs):
        super(BWCFR, self).__init__(args, mlflow_logger)

        self.q_trunc = args.repr_net.q_trunc

    def prepare_train_data(self, data_dict: dict):
        # Scaling train data
        cov_f = self.cov_scaler.fit_transform(data_dict['cov_f'].reshape(-1, self.dim_cov))
        out_f = self.out_scaler.fit_transform(data_dict['out_f'].reshape(-1, 1))

        # Torch tensors
        cov_f, treat_f, out_f = self.prepare_tensors(cov_f, data_dict['treat_f'], out_f)
        prop_pred_cov = torch.tensor(data_dict['prop_pred_cov'].reshape(-1, 1)).float()

        self.hparams.dataset.n_samples_train = cov_f.shape[0]

        return cov_f, treat_f, out_f, prop_pred_cov

    def get_train_dataloader(self, cov_f, treat_f, out_f, prop_pred_cov):
        training_data = TensorDataset(cov_f, treat_f, out_f, prop_pred_cov)
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True,
                                      generator=torch.Generator(device=self.device))
        return train_dataloader


    def fit(self, train_data_dict: dict, log: bool):
        cov_f, treat_f, out_f, prop_pred_cov = self.prepare_train_data(train_data_dict)
        train_dataloader = self.get_train_dataloader(cov_f, treat_f, out_f, prop_pred_cov)
        optimizer = self.get_optimizer()

        for step in tqdm(range(self.num_epochs)) if log else range(self.num_epochs):
            cov_f, treat_f, out_f, prop_pred_cov = next(iter(train_dataloader))
            optimizer.zero_grad()

            repr_f = self.repr_nn(cov_f)

            with torch.no_grad():
                trunc_mask = (prop_pred_cov > self.q_trunc) & (prop_pred_cov < 1.0 - self.q_trunc)
                weights = trunc_mask / (treat_f * prop_pred_cov + (1.0 - treat_f) * (1.0 - prop_pred_cov) + 1e-9)

            loss, log_dict = self.forward_train(repr_f, treat_f, out_f, cov_f, weights, prefix='train')

            loss.backward()
            optimizer.step()

            if step % 50 == 0 and log:
                self.mlflow_logger.log_metrics(log_dict, step=step)
