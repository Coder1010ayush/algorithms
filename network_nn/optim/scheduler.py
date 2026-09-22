from typing import Callable, Optional

import numpy as np

from network_nn.optim.optimizer import Optimizer


class LRScheduler:
    """Base scheduler: call `step()` once per epoch (or iteration) after `optimizer.step()`."""

    def __init__(self, optimizer: Optimizer, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.min_lr = min_lr
        self.last_step = 0

    def get_lr(self, step: int) -> float:
        raise NotImplementedError

    def step(self) -> float:
        self.last_step += 1
        self.optimizer.lr = max(self.min_lr, self.get_lr(self.last_step))
        return self.optimizer.lr


class ConstantLR(LRScheduler):
    def get_lr(self, step):
        return self.base_lr


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, min_lr: float = 0.0):
        super().__init__(optimizer, min_lr)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self, step):
        return self.base_lr * self.gamma ** (step // self.step_size)


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float = 0.95, min_lr: float = 0.0):
        super().__init__(optimizer, min_lr)
        self.gamma = gamma

    def get_lr(self, step):
        return self.base_lr * self.gamma**step


class LinearLR(LRScheduler):
    def __init__(self, optimizer, total_steps: int, final_lr: float = 0.0):
        super().__init__(optimizer, 0.0)
        self.total_steps, self.final_lr = total_steps, final_lr

    def get_lr(self, step):
        frac = min(1.0, step / self.total_steps)
        return self.base_lr * (1 - frac) + self.final_lr * frac


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, total_steps: int, min_lr: float = 0.0):
        super().__init__(optimizer, min_lr)
        self.total_steps = total_steps

    def get_lr(self, step):
        frac = min(1.0, step / self.total_steps)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * frac))


class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_steps: int, power: float = 2.0, min_lr: float = 0.0):
        super().__init__(optimizer, min_lr)
        self.total_steps, self.power = total_steps, power

    def get_lr(self, step):
        frac = min(1.0, step / self.total_steps)
        return self.base_lr * (1 - frac) ** self.power


class CyclicLR(LRScheduler):
    """Triangular cycle between min_lr and max_lr with period 2 * step_size."""

    def __init__(self, optimizer, step_size: int, max_lr: float, min_lr: float = 0.0):
        super().__init__(optimizer, min_lr)
        self.step_size, self.max_lr = step_size, max_lr

    def get_lr(self, step):
        cycle = np.floor(1 + step / (2 * self.step_size))
        x = np.abs(step / self.step_size - 2 * cycle + 1)
        return self.min_lr + (self.max_lr - self.min_lr) * max(0.0, 1 - x)


class WarmupLR(LRScheduler):
    """Linear warmup for `warmup_steps`, then delegates to `after` (another scheduler) if given."""

    def __init__(self, optimizer, warmup_steps: int, after: Optional[LRScheduler] = None):
        super().__init__(optimizer, 0.0)
        self.warmup_steps, self.after = warmup_steps, after

    def get_lr(self, step):
        if step <= self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        return self.after.get_lr(step - self.warmup_steps) if self.after else self.base_lr


class ReduceLROnPlateau:
    """Multiply lr by `factor` when the monitored metric stops improving for `patience` steps."""

    def __init__(self, optimizer: Optimizer, factor: float = 0.1, patience: int = 10, min_lr: float = 0.0, mode: str = "min"):
        self.optimizer, self.factor, self.patience, self.min_lr = optimizer, factor, patience, min_lr
        self.better: Callable[[float, float], bool] = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self.best: Optional[float] = None
        self.bad_steps = 0

    def step(self, metric: float) -> float:
        if self.best is None or self.better(metric, self.best):
            self.best, self.bad_steps = metric, 0
        else:
            self.bad_steps += 1
            if self.bad_steps > self.patience:
                self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.factor)
                self.bad_steps = 0
        return self.optimizer.lr
