import os.path
import pickle

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()
SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SETUP_DIR, "stan")
MODEL_TARGET_DIR = os.path.join("bpl", "stan_model")


class BPyCmd(build_py):
    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            compile_stan_models(target_dir)

        build_py.run(self)


class DevCmd(develop):
    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.setup_path, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            compile_stan_models(target_dir)

        develop.run(self)


def compile_stan_models(target_dir, model_dir=MODEL_DIR):
    """Pre-compile the stan models that are used by the module."""
    from pystan import StanModel

    names = ["simple_model.stan", "model_with_prior.stan"]
    targets = ["simple_model.pkl", "prior_model.pkl"]
    for (name, target) in zip(names, targets):
        sm = StanModel(file=os.path.join(model_dir, name))
        with open(os.path.join(target_dir, target), "wb") as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)


setup(
    name="bpl",
    version="0.0.1",
    description="Simple Bayesian models for football leagues",
    url="https://github.com/anguswilliams91/FPL",
    author="Angus Williams <anguswilliams91@gmail.com>",
    author_email="anguswilliams91@gmail.com",
    license="GPL-3.0",
    packages=["bpl", "bpl.test"],
    setup_requires=[],
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False,
    include_package_data=True,
    cmdclass={"build_py": BPyCmd, "develop": DevCmd},
    test_suite="nose.collector",
    tests_require=["nose"],
    long_description="""
    A package for fitting Bayesian models to football leagues. Models are reminiscent of Dixon \& Coles
    (1997) , but have been cast into a hierarchical Bayesian form, with the option of team-level covariates in the
    prior.""",
)
