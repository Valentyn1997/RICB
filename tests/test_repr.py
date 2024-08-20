from hydra.experimental import initialize, compose
import logging

from runnables.train import main

logging.basicConfig(level='info')


class TestReprNETs:
    def test_tarnet(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=tarnet",
                                                                 "+repr_net/synthetic_hparams/tarnet/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_cfrnet_mmd(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=cfrnet",
                                                                 "+repr_net/synthetic_hparams/cfrnet/n500/mmd_0_5='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_cfrnet_wass(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=cfrnet",
                                                                 "+repr_net/synthetic_hparams/cfrnet/n500/wass_2_0='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_rcfrnet(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=rcfrnet",
                                                                 "+repr_net/synthetic_hparams/rcfrnet/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_cfrisw(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=cfrisw",
                                                                 "+repr_net/synthetic_hparams/cfrisw/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_bwcfr(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=bwcfr",
                                                                 "+repr_net/synthetic_hparams/bwcfr/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_bnnet(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=bnnet",
                                                                 "+repr_net/synthetic_hparams/bnnet/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])
            results = main(args)
            assert results is not None

    def test_inv_tarnet(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=synthetic",
                                                                 "+repr_net=inv_tarnet",
                                                                 "+repr_net/synthetic_hparams/inv_tarnet/n500='1.0'",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "repr_net.num_epochs=5",
                                                                 "prop_net_cov.num_epochs=5",
                                                                 "cnf_repr.num_epochs=5",
                                                                 "prop_net_repr.num_epochs=5"])

            results = main(args)
            assert results is not None
