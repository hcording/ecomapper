"""
Fixing of random seeds.
"""

import importlib


def set_seed(seed: int) -> None:
    """
    Sets the seed for all used libraries, except PyTorch!
    Use ``set_torch_seed`` to set the PyTorch seed.
    This separation is done because the PyTorch import takes a long time.

    :param seed: Seed to set.
    """
    random = importlib.import_module('random')
    np_random = importlib.import_module('numpy.random')

    random.seed(seed)
    np_random.seed(seed)


def set_torch_seed(seed: int) -> None:
    """
    Sets the seed for the PyTorch and PyTorch Cuda libraries.

    :param seed: Seed to set.
    """
    torch = importlib.import_module('torch')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
