from hydra.experimental import initialize, compose
import logging

from runnables.train_classic import main

logging.basicConfig(level='info')


class TestClassicalMethods:
    def test_knn(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+classic_cate=knn",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "dataset.n_samples_train=500"])
            results = main(args)
            assert results is not None

    def test_causal_forest(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+classic_cate=causal_forest",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "dataset.n_samples_train=500"])
            results = main(args)
            assert results is not None
