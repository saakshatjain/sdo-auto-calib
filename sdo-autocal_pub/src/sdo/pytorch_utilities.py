import logging
import multiprocessing
import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np

_logger = logging.getLogger(__name__)
_dtype = torch.float # this corresponds to float32


def to_tensor(value):
    if not torch.is_tensor(value):
        if type(value) == np.int64:
            value = torch.tensor(float(value))
        elif type(value) == np.float32:
            value = torch.tensor(float(value))
        else:
            value = torch.tensor(value)
    return value


def to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except Exception as e:
            print(e)
            raise TypeError('Cannot convert to Numpy array.')



def set_seed(random_seed=1, deterministic_cuda=True):
    """
    Force runs to be deterministic and reproducible. Note that forcing CUDA to be
    deterministic can have a performance impact.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pass_seed_to_worker(worker_id):
    """
    Given some pytorch DataLoader that is spawning worker forks, this method
    will ensure that they are all given the correct random seed on
    initialization to prevent the following problem:
    https://github.com/pytorch/pytorch/issues/5059
    Keep in mind that pytorch creates and destroys these workers on _every_
    epoch, so we have to be extra careful about setting our random seeds
    so they won't repeat every epoch!
    """
    # Numpy can't have random seeds greater than 2^32 - 1.
    seed = (torch.initial_seed() // (2**32 - 1)) + worker_id
    set_seed(seed)


import multiprocessing  # Ensure this is imported at the start of the script

def create_dataloader(dataset, batch_size, num_dataloader_workers, train, shuffle=True):
    # Dynamically set `num_dataloader_workers` to ensure it's within available CPU limits
    available_workers = max(1, multiprocessing.cpu_count() - 1)
    num_dataloader_workers = min(num_dataloader_workers, available_workers)

    _logger.info('Using {} workers for the {} pytorch DataLoader'.format(
        num_dataloader_workers, 'training' if train else 'testing'))
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_dataloader_workers,
        worker_init_fn=pass_seed_to_worker,
        pin_memory=True
    )
    return loader
