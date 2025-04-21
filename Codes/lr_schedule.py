# lr_schedule.py

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

def get_scheduler(optimizer, mode='step', **kwargs):
    """
    mode: 'step', 'cosine', 'plateau'
    kwargs:
        step_size: for StepLR
        gamma: decay rate
        T_max: for CosineAnnealingLR
        patience: for ReduceLROnPlateau
    """
    if mode == 'step':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 50), gamma=kwargs.get('gamma', 0.1))
    elif mode == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 100))
    elif mode == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', patience=kwargs.get('patience', 10), factor=0.5)
    else:
        raise NotImplementedError(f"Scheduler mode '{mode}' is not supported.")
