import pickle
import pkg_resources


def load_stan_model(model_name):
    """Load precompiled StanModel"""
    model_file = pkg_resources.resource_filename(
        "bpl", "stan_model/{}.pkl".format(model_name)
    )
    with open(model_file, "rb") as f:
        return pickle.load(f)


simple_stan_model = load_stan_model("simple_model")
prior_stan_model = load_stan_model("prior_model")
