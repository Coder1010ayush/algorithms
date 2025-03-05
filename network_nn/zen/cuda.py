#  ------------------------------------ utf-8 encoding ------------------------------
import numpy as np


def device(array, device_id=0):
    try:
        import cupy as cp
        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
            if device_id >= n_devices:
                raise RuntimeError(f"Device {device_id} not found. Available devices: {n_devices}")
            with cp.cuda.Device(device_id):
                return cp.asarray(array)
        except cp.cuda.runtime.CUDARuntimeError:
            return array
    except ImportError:
        return array


def as_numpy_array(x):
    from zen.ctx import Tensor
    if isinstance(x, Tensor):
        x = x.data
    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x


def as_cupy_array(x):
    from zen.ctx import Tensor
    if isinstance(x, Tensor):
        x = x.data
    from torch import is_grad_enabled
    if not is_grad_enabled():
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    else:
        try:
            import cupy as cp
            return cp.asarray(x)
        except ImportError:
            return x
