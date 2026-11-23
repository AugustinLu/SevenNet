"""The rules of the enable_mps option, shared by SevenNetCalculator, Trainer and
sevenn.scripts.inference.inference. Imports nothing of the Apple GPU code, so CPU and
CUDA runs never load it."""

import os
from typing import Optional, Tuple, Union

import torch


def device_type(device) -> Optional[str]:
    """'cuda', 'mps', 'cpu', ... of a device argument; None for 'auto', None, or a
    bare CUDA ordinal, which are passed on unchanged as before."""
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str) and device != 'auto':
        return torch.device(device).type
    return None


def resolve(
    enable_mps: Optional[bool], device: Union[torch.device, str, int, None]
) -> Tuple[Optional[bool], Union[torch.device, str, int, None]]:
    """Check enable_mps and return (enable_mps, device): True with device='auto'
    selects 'mps', True on another device is an error. None stays None here; see
    wanted()."""
    if enable_mps is not None and not isinstance(enable_mps, bool):
        raise ValueError(
            f'enable_mps must be True, False or None, got {enable_mps!r}'
        )
    if enable_mps and isinstance(device, str) and device == 'auto':
        device = 'mps'
    if enable_mps and device_type(device) != 'mps':
        raise ValueError(f"enable_mps=True needs device='mps', got {device}")
    return enable_mps, device


def wanted(enable_mps: Optional[bool]) -> bool:
    """On device 'mps': an explicit argument wins; None means on unless the
    environment variable SEVENNET_ENABLE_MPS is '0'."""
    if enable_mps is not None:
        return enable_mps
    return os.environ.get('SEVENNET_ENABLE_MPS') != '0'


def conflicts(**accelerators: bool) -> None:
    """The Apple GPU cannot be combined with the CUDA accelerators."""
    on = [name for name, value in accelerators.items() if value]
    if on:
        raise ValueError(
            'enable_mps runs on the Apple GPU: it cannot be combined with '
            'cuEquivariance, FlashTP, OpenEquivariance or LunaTP (got '
            + ', '.join(on)
            + ')'
        )
