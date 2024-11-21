"""
PyTorch utilities.
"""

import torch.cuda


def is_running_with_multiple_gpus():
    """
    Checks whether the application is running in a multi-GPU environment.

    :return: ``True`` if CUDA is available and the device count exceeds 1,
        otherwise ``False``.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 1
