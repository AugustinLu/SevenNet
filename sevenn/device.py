import torch

def get_auto_device() -> str:
    """
    Returns the best available device for computation.
    Prioritizes CUDA, then MPS (Apple Silicon), then CPU.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
