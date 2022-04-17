import torch


def move_to_device(x, device):
    """Move x to device.

    Args:
        x (dict or list or Tensor): variable to be moved.
        device (torch.device): device.

    Raises:
        RuntimeError: when x is not a dict or a list or a Tensor.
    """
    assert isinstance(device, torch.device)
    if isinstance(x, dict):
        for key in x:
            move_to_device(x[key], device)
    elif isinstance(x, list):
        for i in range(len(x)):
            move_to_device(x[i], device)
    elif isinstance(x, torch.Tensor):
        x.to(device)
    else:
        raise RuntimeError(f"can not move {type(x)} to {device}")
