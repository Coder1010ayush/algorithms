# -------------------------------------------- utf-8 encoding ------------------------------------------
# this file contains all the major learning rate schedule (basic implementation , may be add more functionality in future.)
import numpy as np
from typing import Literal
class LRScheduler:
    def __init__(self, optimizer, initial_lr, schedule_type:Literal["constant", "step" , "exponential", "cosine" , "linear", "plateau" , "cyclical","polynomial"]="constant", **kwargs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type.lower()
        self.kwargs = kwargs
        self.step_count = 0
        self.min_lr = kwargs.get("min_lr", 0.0)
        self.max_lr = kwargs.get("max_lr", initial_lr)
        self.total_steps = kwargs.get("total_steps", 100)
        self.epsilon = 1e-8
        self.validate_params()
    
    def validate_params(self):
        if self.initial_lr <= 0 or self.min_lr < 0 or self.max_lr <= 0:
            raise ValueError("Learning rates must be positive values.")
        if self.min_lr > self.max_lr:
            raise ValueError("min_lr cannot be greater than max_lr.")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")

    def step(self):
        self.step_count += 1
        new_lr = self._get_lr()
        new_lr = max(self.min_lr, min(new_lr, self.max_lr))  
        self.optimizer.lr = new_lr

    def _get_lr(self):
        if self.schedule_type == "constant":
            return self.initial_lr
        
        elif self.schedule_type == "step":
            step_size = self.kwargs.get("step_size", 10)
            gamma = self.kwargs.get("gamma", 0.1)
            return self.initial_lr * (gamma ** (self.step_count // step_size))
        
        elif self.schedule_type == "exponential":
            gamma = self.kwargs.get("gamma", 0.9)
            return self.initial_lr * (gamma ** self.step_count)
        
        elif self.schedule_type == "linear":
            final_lr = self.kwargs.get("final_lr", 1e-5)
            alpha = min(1.0, self.step_count / self.total_steps)
            return self.initial_lr * (1 - alpha) + final_lr * alpha
        
        elif self.schedule_type == "cosine":
            cos_decay = 0.5 * (1 + np.cos(np.pi * self.step_count / self.total_steps))
            return self.min_lr + (self.initial_lr - self.min_lr) * cos_decay
        
        elif self.schedule_type == "cyclical":
            step_size = self.kwargs.get("step_size", 10)
            cycle = np.floor(1 + self.step_count / (2 * step_size))
            x = np.abs(self.step_count / step_size - 2 * cycle + 1)
            return self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x))
        
        elif self.schedule_type == "polynomial":
            power = self.kwargs.get("power", 2.0)
            return self.initial_lr * ((1 - self.step_count / self.total_steps) ** power)
        
        elif self.schedule_type == "plateau":
            patience = self.kwargs.get("patience", 10)
            factor = self.kwargs.get("factor", 0.1)
            if self.step_count % patience == 0:
                return max(self.min_lr, self.optimizer.lr * factor)
            return self.optimizer.lr
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
