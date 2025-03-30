import numpy as np
from tensor import Tensor  # Assuming you have a custom Tensor class

class Initializer:
    def __init__(self):
        pass
    
    @staticmethod
    def _get_tensor(tensor_or_shape):
        if isinstance(tensor_or_shape, (tuple, list)):
            return Tensor(data=np.empty(tensor_or_shape))
        elif isinstance(tensor_or_shape, Tensor):
            return tensor_or_shape
        else:
            raise TypeError("Input must be a Tensor or a tuple/list representing the shape")
    
    @staticmethod
    def uniform(tensor_or_shape, a=0.0, b=1.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        tensor.data = np.random.uniform(a, b, size=tensor.data.shape)
        return tensor
    
    @staticmethod
    def normal(tensor_or_shape, mean=0.0, std=1.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        tensor.data = np.random.normal(mean, std, size=tensor.data.shape)
        return tensor

    @staticmethod
    def constant(tensor_or_shape, val=0.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        tensor.data.fill(val)
        return tensor
    
    @staticmethod
    def ones(tensor_or_shape):
        tensor = Initializer._get_tensor(tensor_or_shape)
        tensor.data.fill(1.0)
        return tensor
    
    @staticmethod
    def zeros(tensor_or_shape):
        tensor = Initializer._get_tensor(tensor_or_shape)
        tensor.data.fill(0.0)
        return tensor

    @staticmethod
    def xavier_uniform(tensor_or_shape, gain=1.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        fan_in, fan_out = Initializer._calculate_fan_in_out(tensor.data.shape)
        limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
        tensor.data = np.random.uniform(-limit, limit, size=tensor.data.shape)
        return tensor
    
    @staticmethod
    def xavier_normal(tensor_or_shape, gain=1.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        fan_in, fan_out = Initializer._calculate_fan_in_out(tensor.data.shape)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        tensor.data = np.random.normal(0.0, std, size=tensor.data.shape)
        return tensor
    
    @staticmethod
    def kaiming_uniform(tensor_or_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        tensor = Initializer._get_tensor(tensor_or_shape)
        fan = Initializer._calculate_fan(tensor.data.shape, mode)
        gain = np.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == 'leaky_relu' else np.sqrt(2.0)
        limit = gain * np.sqrt(3.0 / fan)
        tensor.data = np.random.uniform(-limit, limit, size=tensor.data.shape)
        return tensor

    @staticmethod
    def kaiming_normal(tensor_or_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        tensor = Initializer._get_tensor(tensor_or_shape)
        fan = Initializer._calculate_fan(tensor.data.shape, mode)
        gain = np.sqrt(2.0 / (1 + a ** 2)) if nonlinearity == 'leaky_relu' else np.sqrt(2.0)
        std = gain / np.sqrt(fan)
        tensor.data = np.random.normal(0.0, std, size=tensor.data.shape)
        return tensor
    
    @staticmethod
    def orthogonal(tensor_or_shape, gain=1.0):
        tensor = Initializer._get_tensor(tensor_or_shape)
        flat_shape = (tensor.data.shape[0], np.prod(tensor.data.shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        tensor.data = (u if u.shape == flat_shape else v).reshape(tensor.data.shape) * gain
        return tensor
    
    @staticmethod
    def _calculate_fan_in_out(shape):
        if len(shape) < 2:
            raise ValueError("Tensor shape must have at least two dimensions")
        fan_in = shape[1] if len(shape) > 1 else shape[0]
        fan_out = shape[0]
        return fan_in, fan_out
    
    @staticmethod
    def _calculate_fan(shape, mode):
        fan_in, fan_out = Initializer._calculate_fan_in_out(shape)
        return fan_in if mode == 'fan_in' else fan_out

shape = (128, 64)
Initializer.xavier_uniform(shape)
x = Initializer.kaiming_normal((64, 32), a=0.01, mode='fan_in', nonlinearity='relu')
print(x)