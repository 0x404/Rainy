"""Helper function"""
import torch


def move_to_device(variable, device):
    """Move variable to device recursively.

    Args:
        variable (dict or list or Tensor): variable to be moved.
        device (torch.device): device.

    Raises:
        RuntimeError: when variable is not a dict or a list or a Tensor.

    Return:
        dict or list or Tensor on specified device.
    """
    assert isinstance(device, torch.device)
    if isinstance(variable, dict):
        return {k: move_to_device(v, device) for k, v in variable.items()}
    if isinstance(variable, list):
        return [move_to_device(v, device) for v in variable]
    if isinstance(variable, torch.Tensor):
        return variable.to(device)
    raise RuntimeError(f"can not move {type(variable)} to {device}")
