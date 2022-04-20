"""Helper function"""
import torch


def move_to_device(x, device):
    """Move x to device recursively.

    Args:
        x (dict or list or Tensor): variable to be moved.
        device (torch.device): device.

    Raises:
        RuntimeError: when x is not a dict or a list or a Tensor.

    Return:
        dict or list or Tensor on specified device.
    """
    assert isinstance(device, torch.device)
    if isinstance(x, dict):
        return {k : move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        raise RuntimeError(f"can not move {type(x)} to {device}")
