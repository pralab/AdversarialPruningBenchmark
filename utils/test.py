import io
import torch
import pickle
import contextlib
import autoattack
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data import DataLoader
from HOFMN.src.attacks.fmn import FMN
from HOFMN.src.ho_fmn.ho_fmn import HOFMN


def test_model_aa(model, ds, data_dir, device):
    # define transform set
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)

    # autoattack base settings
    norm = 'Linf'
    version = 'standard'
    epsilon = 0.031
    batch_size = 500
    n_ex = 10000

    # define dataset
    if ds == 'CIFAR10':
        item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    elif ds == 'SVHN':
        item = datasets.SVHN(root=data_dir, split='test', transform=transform_chain, download=True)
    else:
        raise KeyError('Wrong dataset: choose between CIFAR10 and SVHN.')

    # define test_loader
    test_loader = data.DataLoader(item, batch_size=batch_size, shuffle=False, num_workers=0)

    # create test set
    set_aux = [x for (x, y) in test_loader]
    x_test = torch.cat(set_aux, 0)
    set_aux = [y for (x, y) in test_loader]
    y_test = torch.cat(set_aux, 0)

    # set attack
    model.eval()
    model.to(device)
    adversary = autoattack.AutoAttack(model, norm=norm, eps=epsilon, version=version, device=device)
    # capture output
    output_stream = io.StringIO()

    # Step 2: Redirect the standard output to the string IO stream
    with contextlib.redirect_stdout(output_stream):
        adversary.run_standard_evaluation(x_test[:n_ex].to(device), y_test[:n_ex].to(device), bs=batch_size)

    # Step 3: Retrieve the output
    output = output_stream.getvalue()

    # Step 4: Parse the output to extract initial and robust accuracy
    clean_acc = None
    rob_acc = None

    # Iterate over each line in the output
    for line in output.splitlines():
        if "initial accuracy" in line:
            clean_acc = float(line.split(":")[1].strip().strip('%'))
        elif "robust accuracy:" in line:
            rob_acc = float(line.split(":")[1].split('%')[0].strip())

    # Close the string IO stream
    output_stream.close()

    return clean_acc, rob_acc


def test_model_hofmn(model, model_key, ds, data_dir, device, save_dist_path, loss='DLR', optimizer='SGD', scheduler='CALR', get_distances=True):
    # HO-FMN configuration
    steps = 100  # The number of FMN attack iterations
    trials = 32  # Number of HO optimization trials
    tuning_bs = 64  # Batch size for the tuning

    # define dataset
    if ds == 'CIFAR10':
        item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
    elif ds == 'SVHN':
        item = datasets.SVHN(root=data_dir, split='test', transform=transform_chain, download=True)
    else:
        raise KeyError('Wrong dataset: choose between CIFAR10 and SVHN.')

    # define test_loader
    dataloader = data.DataLoader(item, batch_size=tuning_bs, shuffle=False, num_workers=0)

    ho_fmn = HOFMN(
        model=model,
        dataloader=dataloader,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        steps=steps,
        trials=trials,
        verbose=True,
        device=device
    )

    # Start the tuning process
    best_parameters = ho_fmn.tune()
    print(f"Best parameters:\n{best_parameters}")

    # Compute the samples used for the tuning
    tuning_trials = trials
    tuning_samples = tuning_bs * tuning_trials

    attack_bs = 128  # Attack batch size
    attack_steps = 200  # Attack steps

    # define new set of points #TODO check
    subset_indices = list(range(tuning_samples, tuning_samples + attack_bs))
    dataset_frac = torch.utils.data.Subset(item, subset_indices)

    dataloader = DataLoader(
        dataset=dataset_frac,
        batch_size=attack_bs,
        shuffle=False
    )

    # Extract the optimizer and scheduler config from the best params dictionary
    optimizer_config, scheduler_config = ho_fmn.parametrization_to_configs(best_parameters,
                                                                           batch_size=attack_bs,
                                                                           steps=attack_steps)

    model.eval()
    model.to(device)

    tuned_attack = FMN(
        model=model,
        steps=attack_steps,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        device=device
    )

    total = 0.0
    distances = []

    for (images, labels) in dataloader:
        total += labels.shape[0]
        print(total)
        if total < 8500:
            tuned_best_adv = tuned_attack.forward(images=images, labels=labels)
            norms = (tuned_best_adv.cpu() - images.cpu()).flatten(1).norm(torch.inf, dim=1)
            distances.extend(norms)

    rob_acc = (distances > 8 / 255).float().mean().item()

    if get_distances:
        path_name = f'{save_dist_path}/{model_key}.pickle'
        with open(path_name, 'wb') as handle:
            pickle.dump(distances, handle)
        print(f'Distances saved at {save_dist_path}')
        return rob_acc, distances
    else:
        return rob_acc


def benchmark(model, model_key, data_dir, save_dist_path, device):
    # define dataset
    if 'CIFAR10' in model_key:
        ds = 'CIFAR10'
    elif 'SVHN' in model_key:
        ds = 'SVHN'
    else:
        raise KeyError('Wrong dataset: choose between CIFAR10 and SVHN.')

    # get aa robustness
    clean_acc, rob_acc_aa = test_model_aa(model, ds, data_dir, device)
    rob_acc_hofmn = test_model_hofmn(model, model_key, ds, data_dir, device, save_dist_path=save_dist_path, loss='DLR', optimizer='SGD', scheduler='CALR', get_distances=True)

    # print data ready for claim
    print('Results ready for claim! Copy and paste the following data:\n')
    # model key and gdrive id
    print(f'"{model_key}" : ["your_gdrive_link_id", "extension"]\n')
    print(f'Clean Accuracy: {clean_acc}')
    print(f'Robust Accuracy from AA: {rob_acc_aa}')
    print(f'Robust Accuracy from HO-FMN: {rob_acc_hofmn}')


