import logging
import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit

from src.models.msm import MSM
from src.models.utils import subset_by_indices
from src import ROOT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of dataset
    torch.set_default_device(args.exp.device)
    seed_everything(args.exp.seed)
    dataset = instantiate(args.dataset, _recursive_=True)
    data_dicts = dataset.get_data()
    if not args.dataset.collection:
        data_dicts = [data_dicts]
    if args.dataset.dataset_ix is not None:
        data_dicts = [data_dicts[args.dataset.dataset_ix]]
        specific_ix = True
    else:
        specific_ix = False

    for ix, data_dict in enumerate(data_dicts):

        # ================= Train / test split & Mlflow init =================
        experiment_name = f'{args.repr_net.name}/{args.dataset.name}/new'
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name,
                                     tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

        args.dataset.dataset_ix = ix if not specific_ix else args.dataset.dataset_ix
        if args.dataset.train_test_splitted:
            train_data_dict, test_data_dict = data_dict
        else:
            ss = ShuffleSplit(n_splits=1, random_state=2 * args.exp.seed, test_size=args.dataset.test_size)
            train_index, test_index = list(ss.split(train_data_dict['cov_f']))[0]
            train_data_dict, val_data_dict = subset_by_indices(data_dict, train_index), subset_by_indices(data_dict, test_index)

        # ================= Train-validation split for downstream models =================

        ss = ShuffleSplit(n_splits=1, random_state=args.exp.seed, test_size=args.dataset.test_size)
        ttrain_index, val_index = list(ss.split(train_data_dict['cov_f']))[0]

        ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, ttrain_index), \
            subset_by_indices(train_data_dict, val_index)

        # ================= Fitting representation net =================
        if args.repr_net.has_prop_net_cov:  # For BWCFR
            prop_net_cov = instantiate(args.prop_net_cov, args, mlflow_logger, 'cov', _recursive_=True)

            # Finetuning for the first split
            if args.prop_net_cov.tune_hparams and ix == 0:
                prop_net_cov.finetune(ttrain_data_dict, {'cpu': 0.4, 'gpu': 0.0}, val_data_dict)

            # Training
            logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
            prop_net_cov.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_cov = prop_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

            # Getting propensity weights
            for data_dict in [train_data_dict, ttrain_data_dict, val_data_dict, test_data_dict]:
                data_dict['prop_pred_cov'] = prop_net_cov.get_prop_predictions(data_dict=data_dict)

        repr_net = instantiate(args.repr_net, args, mlflow_logger, _recursive_=True)

        # Finetuning for the first split
        if args.repr_net.tune_hparams and ix == 0:
            repr_net.finetune(train_data_dict, {'cpu': 0.4, 'gpu': 0.0})

        # Training
        logger.info(f'Fitting a representation net for sub-dataset {args.dataset.dataset_ix}.')
        repr_net.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

        # Evaluation
        results_in = repr_net.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        results_out = repr_net.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
        logger.info(f'In-sample performance: {results_in}. Out-sample performance: {results_out}')

        repr_net.evaluate_policy(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        repr_net.evaluate_policy(data_dict=test_data_dict, log=args.exp.logging, prefix='out')

        # Inferring representations & predictions
        for data_dict in [train_data_dict, ttrain_data_dict, val_data_dict, test_data_dict]:
            data_dict['repr_f'] = repr_net.get_representations(data_dict=data_dict)
            data_dict['mu_pred0_repr'], data_dict['mu_pred1_repr'] = repr_net.get_outcomes(data_dict=data_dict)

        # ================= Estimating sensitivity parameters of MSM =================
        if not args.repr_net.has_prop_net_cov:
            prop_net_cov = instantiate(args.prop_net_cov, args, mlflow_logger, 'cov', _recursive_=True)

            # Finetuning for the first split
            if args.prop_net_cov.tune_hparams and ix == 0:
                prop_net_cov.finetune(ttrain_data_dict, {'cpu': 0.4, 'gpu': 0.0}, val_data_dict)

            # Training
            logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
            prop_net_cov.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_cov = prop_net_cov.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance propensity cov: {results_out_cov}')

        if args.repr_net.has_prop_net_repr:  # For CFR-ISW
            prop_net_repr = repr_net.prop_net_repr
        else:
            prop_net_repr = instantiate(args.prop_net_repr, args, mlflow_logger, 'repr', _recursive_=True)

            # Finetuning for the first split
            if args.prop_net_repr.tune_hparams and ix == 0:
                prop_net_repr.finetune(ttrain_data_dict, {'cpu': 0.4, 'gpu': 0.0}, val_data_dict)

            # Training
            logger.info(f'Fitting propensity net for sub-dataset {args.dataset.dataset_ix}.')
            prop_net_repr.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

            # Evaluation
            results_out_repr = prop_net_repr.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
            logger.info(f'Out-sample performance propensity repr: {results_out_repr}')

        # Estimating Gamma
        for data_dict in [train_data_dict, ttrain_data_dict, val_data_dict, test_data_dict]:
            data_dict['prop_pred_cov'] = prop_net_cov.get_prop_predictions(data_dict=data_dict)
            data_dict['prop_pred_repr'] = prop_net_repr.get_prop_predictions(data_dict=data_dict)
            for delta in args.msm.delta:
                data_dict[f'gammas_{delta}'] = MSM.calculate_gammas(data_dict, reference_data_dict=train_data_dict, delta=delta)

        if args.exp.logging:
            for delta in args.msm.delta:
                mlflow_logger.log_metrics({f'in_gamma_median_{delta}': np.median(train_data_dict[f'gammas_{delta}']),
                                           f'out_gamma_median_{delta}': np.median(test_data_dict[f'gammas_{delta}'])})
                logger.info(f'In-sample Gamma median for delta = {delta}: {np.median(train_data_dict[f"gammas_{delta}"])}; '
                            f'out-sample Gamma median for delta = {delta}: {np.median(test_data_dict[f"gammas_{delta}"])}')

        # ================= Estimating bounds of MSM =================
        cnf_repr = instantiate(args.cnf_repr, args, mlflow_logger, _recursive_=True)
        # Finetuning for the first split
        if args.cnf_repr.tune_hparams and ix == 0:
            cnf_repr.finetune(ttrain_data_dict, {'cpu': 0.4, 'gpu': 0.0}, val_data_dict)

        # Training
        logger.info(f'Fitting CNF for sub-dataset {args.dataset.dataset_ix}.')
        cnf_repr.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

        # Inferring bounds
        for data_dict in [train_data_dict, ttrain_data_dict, val_data_dict, test_data_dict]:
            for delta in args.msm.delta:
                data_dict[f'mu_pred0_PSM_bounds_{delta}'], data_dict[f'mu_pred1_PSM_bounds_{delta}'] = \
                    cnf_repr.get_bounds(data_dict, delta, n_samples=args.msm.n_samples)

        # Evaluation
        results_in = cnf_repr.evaluate(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        results_out = cnf_repr.evaluate(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
        logger.info(f'In-sample performance CNF: {results_in}, Out-sample performance CNF: {results_out}')

        pehe_in = cnf_repr.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in',
                                         n_samples=args.msm.n_samples)
        pehe_out = cnf_repr.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out',
                                          n_samples=args.msm.n_samples)
        logger.info(f'In-sample PEHE CNF: {pehe_in}, Out-sample PEHE CNF: {pehe_out}')

        for delta in args.msm.delta:
            cnf_repr.evaluate_policy(data_dict=train_data_dict, delta=delta, log=args.exp.logging, prefix='in')
            cnf_repr.evaluate_policy(data_dict=test_data_dict, delta=delta, log=args.exp.logging, prefix='out')

        mlflow_logger.log_hyperparams(args) if args.exp.logging else None

        if args.exp.logging and args.exp.save_results:
            save_dir = f'{ROOT_PATH}/mlruns/{mlflow_logger.experiment_id}/{mlflow_logger.run_id}/artifacts'
            np.save(f'{save_dir}/train_data_dict.npy', train_data_dict)
            np.save(f'{save_dir}/test_data_dict.npy', test_data_dict)

            torch.save(repr_net, f'{save_dir}/repr_net.pt')
            torch.save(cnf_repr, f'{save_dir}/cnf_repr.pt')
            torch.save(prop_net_repr, f'{save_dir}/prop_net_repr.pt')
            torch.save(prop_net_cov, f'{save_dir}/prop_net_cov.pt')

        mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return results_in, results_out
    # return {
    #     'models': {'repr_net': repr_net, 'cnf_repr': cnf_repr, 'prop_net_repr': prop_net_repr, 'prop_net_cov': prop_net_cov},
    #     'data_dicts': {'train_data_dict': train_data_dict, 'test_data_dict': test_data_dict}
    # }


if __name__ == "__main__":
    main()
