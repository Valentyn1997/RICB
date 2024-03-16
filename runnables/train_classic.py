import logging
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from sklearn.model_selection import ShuffleSplit

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
        experiment_name = f'{args.cate_estimator.name}/{args.dataset.name}/new'
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri) if args.exp.logging else None

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

        ttrain_data_dict, val_data_dict = subset_by_indices(train_data_dict, ttrain_index), subset_by_indices(train_data_dict, val_index)

        cate_estimator = instantiate(args.cate_estimator, args, mlflow_logger, _recursive_=True)

        # Finetuning for the first split
        if args.cate_estimator.tune_hparams and ix == 0:
            cate_estimator.finetune(train_data_dict, {'cpu': 0.4, 'gpu': 0.0})

        # Training
        logger.info(f'Fitting a CATE estimator for sub-dataset {args.dataset.dataset_ix}.')
        cate_estimator.fit(train_data_dict=ttrain_data_dict, log=args.exp.logging)

        # Evaluation
        results_in = cate_estimator.evaluate_pehe(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        results_out = cate_estimator.evaluate_pehe(data_dict=test_data_dict, log=args.exp.logging, prefix='out')
        logger.info(f'In-sample performance: {results_in}. Out-sample performance: {results_out}')

        cate_estimator.evaluate_policy(data_dict=train_data_dict, log=args.exp.logging, prefix='in')
        cate_estimator.evaluate_policy(data_dict=test_data_dict, log=args.exp.logging, prefix='out')

        mlflow_logger.log_hyperparams(args) if args.exp.logging else None

        mlflow_logger.experiment.set_terminated(mlflow_logger.run_id) if args.exp.logging else None

    return {
        'models': {'cate_estimator': cate_estimator},
        'data_dicts': {'train_data_dict': train_data_dict, 'test_data_dict': test_data_dict}
    }


if __name__ == "__main__":
    main()
