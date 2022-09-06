"""
Plotting module.
"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

def plot_constrained_state_history(state_history: OrderedDict) -> None:
    """Plot the state history.
    Args:
        state_history (OrderedDict): The state history to plot.
    """
    # Unpack values
    iters, loss_history = zip(
        *[(iter_num, _["cmp"].loss.item()) for iter_num, _ in state_history.items()]
    )
    mult_history = np.stack(
        [_["dual"][0].data.numpy() for _ in state_history.values()]
    )
    _ = np.stack(
        [_["cmp"].ineq_defect.data.numpy() for _ in state_history.values()]
    )

    # Plot
    _, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(15, 3))

    ax0.plot(iters, mult_history)
    ax0.set_title("Multipliers")

    ax1.plot(iters, mult_history, alpha=0.6)
    ax1.axhline(0.0, c="gray", alpha=0.2)
    ax1.set_title("Defects")

    ax2.plot(iters, loss_history)
    ax2.set_title("Objective")

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    _ = [_.semilogx() for _ in (ax0, ax1, ax2)]
    plt.show()

def plot_unconstrained_state_history(state_history: OrderedDict) -> None:
    """Plots loss of unconstrained model.
    Args:
        state_history (OrderedDict):
    """
    iters, loss_history = zip(
        *[(iter_num, _["loss"]) for iter_num, _ in state_history.items()]
    )
    plt.figure(figsize=(5, 3))
    plt.plot(iters, loss_history)
    plt.title("Unconstrained Model Objective")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.semilogx()
    plt.show()
