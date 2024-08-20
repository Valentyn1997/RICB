from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestTarNET:
    def test_tarnet(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=tarnet",
                                                                 "+repr_net/synthetic_hparams/tarnet/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=10",
                                                                 "prop_net_cov.num_epochs=10",
                                                                 "cnf_repr.num_epochs=10",
                                                                 "prop_net_repr.num_epochs=10"])
            results_1, results_2 = main(args), main(args)
            assert results_1 == results_2