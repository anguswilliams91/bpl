import pickle
import pkg_resources


def load_stan_model(model_name):
    """Load precompiled StanModel"""
    model_file = pkg_resources.resource_filename(
        "bpl", "stan_model/{}.pkl".format(model_name)
    )
    with open(model_file, "rb") as f:
        return pickle.load(f)


model_without_covariates = load_stan_model("without_covariates")
model_with_covariates = load_stan_model("with_covariates")
