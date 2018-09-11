try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False

import numpy as np


def check_matplotlib(foo):
    # check if matplotlib is installed
    def foo_check(*args, **kwargs):
        if has_matplotlib:
            return foo(*args, **kwargs)
        else:
            raise ImportError("Matplotlib is not available.")

    return foo_check


@check_matplotlib
def plot_score_grid(prob, home_team, away_team):
    # plot a heat map of probabilities
    fig, ax = plt.subplots()
    im = ax.imshow(prob)
    fig.colorbar(im, ax=ax, label="Result probability")
    most_likely = np.unravel_index(prob.argmax(), prob.shape)
    ax.scatter([most_likely[1]], [most_likely[0]], marker="x", s=100, c="k")
    ax.set_ylabel("{} goals (home)".format(home_team))
    ax.set_xlabel("{} goals (away)".format(away_team))
    fig.tight_layout()
    return fig
