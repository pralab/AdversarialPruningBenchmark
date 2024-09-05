import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

mpl.rcParams.update({'font.size': 16})


def plot_sec_curve(list_of_distances, names, title, save=False):
    # define scale
    pert_sizes = torch.linspace(0, 0.15, 1000).unsqueeze(1)
    # number of models
    n = len(names)
    # define plot
    figure, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=1000)

    # get accuracy and robustness
    rob_accs = [(torch.tensor(norms) > 8 / 255).float().mean().item() for norms in list_of_distances]
    accs = [(torch.tensor(norms) > 0).float().mean().item() for norms in list_of_distances]
    # put norms on scale
    scaled_norms = [(torch.tensor(norms) > pert_sizes).float().mean(dim=1) for norms in list_of_distances]

    # plot curves
    for i in range(0, n):
        lab = names[i] + '-{:0.2f}'.format(rob_accs[i] * 100)
        ax.plot(pert_sizes, scaled_norms[i], label=lab)

    ax.axvline(x=8 / 255, color='#EE004D', linewidth=1, linestyle='--')
    ax.grid(True)
    ax.legend()

    ax.set_xlim(0.0, 0.15)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Perturbation norm')
    ax.set_ylabel('Robust Accuracy')
    ax.set_title(title)
    plt.savefig(title)
    plt.show()


