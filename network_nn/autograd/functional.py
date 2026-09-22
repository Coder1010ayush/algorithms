"""Functional API: every differentiable operation as a plain function on Tensors."""

from network_nn.autograd import activation as _act
from network_nn.autograd import conv as _conv
from network_nn.autograd import elementwise as _ew
from network_nn.autograd import linalg as _la
from network_nn.autograd import loss as _loss
from network_nn.autograd import norm as _norm
from network_nn.autograd import pool as _pool
from network_nn.autograd import reduce as _red
from network_nn.autograd import shape as _shape

# elementwise
add = _ew.Add.apply
sub = _ew.Sub.apply
mul = _ew.Mul.apply
div = _ew.Div.apply
pow = _ew.Pow.apply
neg = _ew.Neg.apply
exp = _ew.Exp.apply
log = _ew.Log.apply
sqrt = _ew.Sqrt.apply
abs = _ew.Abs.apply
sin = _ew.Sin.apply
cos = _ew.Cos.apply
tan = _ew.Tan.apply
clip = _ew.Clip.apply
masked_fill = _ew.MaskedFill.apply
where = _ew.Where.apply
maximum = _ew.Maximum.apply

# reductions
sum = _red.Sum.apply
mean = _red.Mean.apply
var = _red.Var.apply
std = _red.Std.apply
max = _red.Max.apply
min = _red.Min.apply

# shape
reshape = _shape.Reshape.apply
permute = _shape.Permute.apply
squeeze = _shape.Squeeze.apply
unsqueeze = _shape.Unsqueeze.apply
broadcast_to = _shape.BroadcastTo.apply
getitem = _shape.GetItem.apply
pad = _shape.Pad.apply


def stack(tensors, axis=0):
    return _shape.Stack.apply(*tensors, axis=axis)


def concat(tensors, axis=0):
    return _shape.Concat.apply(*tensors, axis=axis)


cat = concat


def transpose(x, axis0=None, axis1=None):
    if axis0 is None:
        return x.transpose()
    return x.transpose(axis0, axis1)


def flatten(x, start_dim=1):
    return x.flatten(start_dim)


# linear algebra
matmul = _la.MatMul.apply

# activations
relu = _act.ReLU.apply
leaky_relu = _act.LeakyReLU.apply
sigmoid = _act.Sigmoid.apply
tanh = _act.Tanh.apply
swish = _act.Swish.apply
silu = swish
gelu = _act.GELU.apply
softmax = _act.Softmax.apply
log_softmax = _act.LogSoftmax.apply

# losses
mse_loss = _loss.MSELoss.apply
binary_cross_entropy = _loss.BCELoss.apply
binary_cross_entropy_with_logits = _loss.BCEWithLogitsLoss.apply
cross_entropy = _loss.CrossEntropyLoss.apply
huber_loss = _loss.HuberLoss.apply

# normalisation (affine parameters are applied by the nn layers)
normalize = _norm.Normalize.apply


def layer_norm(x, eps=1e-5):
    return normalize(x, -1, eps)


def batch_norm(x, eps=1e-5):
    axes = tuple(i for i in range(x.ndim) if i != 1)
    return normalize(x, axes, eps)


# convolution and pooling
def conv1d(x, weight, stride=1, padding=0):
    return _conv.ConvNd.apply(x, weight, stride=stride, padding=padding)


conv2d = conv1d
conv3d = conv1d


def max_pool1d(x, kernel_size, stride=None):
    return _pool.MaxPoolNd.apply(x, kernel_size=kernel_size, stride=stride)


max_pool2d = max_pool1d
max_pool3d = max_pool1d


def avg_pool1d(x, kernel_size, stride=None):
    return _pool.AvgPoolNd.apply(x, kernel_size=kernel_size, stride=stride)


avg_pool2d = avg_pool1d
avg_pool3d = avg_pool1d


def global_avg_pool(x):
    return mean(x, tuple(range(2, x.ndim)))


def dropout(x, p=0.5, training=True, rng=None):
    if not training or p == 0.0:
        return x
    from network_nn.tensor import Tensor
    import numpy as np

    rng = np.random.default_rng() if rng is None else rng
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return mul(x, Tensor(mask.astype(x.dtype)))
